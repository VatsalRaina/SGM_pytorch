#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

from utilities import text_to_array
from models import SimilarityGridModel

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Specify the initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=1.0, help='Specify the dropout keep probability')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
parser.add_argument('--n_samples', type=int, default=1, help='Specify the number of negative samples to take')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--num_topics', type=int, default=379, help='Specify the number of unique topics in training')
parser.add_argument('--vocab_size', type=int, default=62416, help='Number of words in vocabulary')
parser.add_argument('--embd_dim', type=int, default=400, help='Dimensionality for word embeddings')
parser.add_argument('--img_width', type=int, default=180, help='Width of resized similarity grid')
parser.add_argument('--img_height', type=int, default=180, help='Height of resized similarity grid')
parser.add_argument('--train_resps_path', type=str, help='Load path to training responses as text')
parser.add_argument('--valid_resps_path', type=str, help='Load path to vaidation responses as text')
parser.add_argument('--wlist_path', type=str, help='Load path to list of word indices')
parser.add_argument('--unique_prompts_path', type=str, help='Load path to unique prompts as text')
parser.add_argument('--unique_prompts_distribution_path', type=str, help='Load path to distribution of unique prompts')
parser.add_argument('--train_prompts_idxs_path', type=str, help='Load path to training data unique prompt indices (for dynamic shuffling)')
parser.add_argument('--valid_prompts_idxs_path', type=str, help='Load path to valid data unique prompt indices (for dynamic shuffling)')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')


# Set device
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _shuffle(p_id, r, r_len, topics_dist, NUM_TOPICS):
    # Dynamic shuffling in order to generate negative samples
    bs = list(p_id.size())[0]
    y_true_first = np.ones(bs)
    y_true_second = np.zeros(bs)
    y_true = np.concatenate(y_true_first, y_true_second)
    y_true = torch.from_numpy(y_true)
    y_true = y_true.float()
    new_p_id = np.random.choice(NUM_TOPICS, bs, p=topics_dist)
    for i in range(bs):
        while (new_p_id[i] == p_id[i]):
            new_p_id[i] = np.random.choice(NUM_TOPICS, 1, p=topics_dist)
    new_p_id = torch.from_numpy(new_p_id)
    new_p_id = new_p_id.int()
    p_id = torch.cat((p_id, new_p_id), 0)
    r = torch.cat((r, r), 0)
    r_len = torch.cat((r_len, r_len), 0)
    return p_id, r, r_len, y_true
    

def _get_prompts(p_id, topics, topics_lens):
    p = torch.index_select(topics, 0, p_id)
    p_len = torch.index_select(topics_lens, 0, p_id)
    return p, p_len

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    torch.manual_seed(args.seed)
    device = get_default_device()

    # We need the lengths as the numpy array is of fixed size but each sentence is of different length
    # so we need to know how many redundant zeros we have in each sample.
    responses_train, responses_train_lens = text_to_array(args.train_resps_path, args.wlist_path)
    responses_valid, responses_valid_lens = text_to_array(args.valid_resps_path, args.wlist_path)
    topics, topics_lens = text_to_array(args.unique_prompts_path, args.wlist_path)

    # For dynamic shuffling to generate the negative samples each epoch, we need to make sure the source
    # and destination prompts are not the same.
    prompts_train_idxs = np.loadtxt(args.train_prompts_idxs_path, dtype=np.int32)
    prompts_valid_idxs = np.loadtxt(args.valid_prompts_idxs_path, dtype=np.int32)
    topics_dist = np.loadtxt(args.unique_prompts_distribution_path, dtype=np.int32)

    responses_train = torch.from_numpy(responses_train)
    responses_train = responses_train.int()
    responses_train_lens = torch.from_numpy(responses_train_lens)
    responses_train_lens = responses_train_lens.int()

    responses_valid = torch.from_numpy(responses_valid)
    responses_valid = responses_valid.int()
    responses_valid_lens = torch.from_numpy(responses_valid_lens)
    responses_valid_lens = responses_valid_lens.int()

    topics = torch.from_numpy(topics)
    topics = topics.int()
    topics_lens = torch.from_numpy(topics_lens)
    topics_lens = topics_lens.int()

    prompts_train_idxs = torch.from_numpy(prompts_train_idxs)
    prompts_train_idxs = prompts_train_idxs.int()
    prompts_valid_idxs = torch.from_numpy(prompts_valid_idxs)
    prompts_valid_idxs = prompts_valid_idxs.int()

    # Store all training dataset in a single wrapped tensor
    train_ds = TensorDataset(prompts_train_idxs, responses_train, responses_train_lens)
    valid_ds = TensorDataset(prompts_valid_idxs, responses_valid, responses_valid_lens)

    # Use DataLoader to handle minibatches easily
    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = args.batch_size, shuffle = False)

    # Construct model
    NUM_TOPICS = args.num_topics
    VOCAB_SIZE = args.vocab_size
    EMBD_DIM = args.embd_dim
    IMG_WIDTH = args.IMG_WIDTH
    IMG_HEIGHT = args.IMG_HEIGHT
    hyperparameters = {'VOCAB_SIZE': VOCAB_SIZE, 'EMBD_DIM': EMBD_DIM, 'IMG_WIDTH': IMG_WIDTH, 'IMG_HEIGHT': IMG_HEIGHT}
    my_model = SimilarityGridModel(hyperparameters)
    my_model = my_model.float()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr = args.learning_rate)

    for epoch in range(args.n_epochs):
        my_model.train()
        total_loss = 0
        counter = 0
        for p_id, r, r_len in train_dl:
            # Perform dynamic shuffling
            p_id, r, r_len, y_true = _shuffle(p_id, r, r_len, topics_dist, NUM_TOPICS)

            # Load the actual prompts using the topics
            p, p_len = _get_prompts(p_id, topics, topics_lens)

            # Forward pass
            y_pred = my_model.forward(p, p_len, r, r_len, args.batch_size*2)
            # Compute loss
            loss = criterion(y_pred, y_true)

            # Zero gradients, backward pass, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            counter += 1
            print(counter)
        trn_loss = total_loss / counter

        # Calculate dev set loss
        total_loss = 0
        counter = 0
        for p_id, r, r_len in valid_dl:
            p_id, r, r_len, y_true = _shuffle(p_id, r, r_len, topics_dist, NUM_TOPICS)
            p, p_len = _get_prompts(p_id, topics, topics_lens)
            y_pred = my_model.forward(p, p_len, r, r_len, args.batch_size*2)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            counter += 1
        valid_loss = total_loss / counter

        # Report results at end of each epoch
        print("Epoch: ", epoch, " Training Loss: ", trn_loss, " Validation Loss: ", valid_loss)

    # Save the model to a file
    file_path = args.save_path+'sgm_seed'+str(args.seed)+'.pt'
    torch.save(my_model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)