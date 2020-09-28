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
parser.add_argument('--train_data_path', type=str, help='Load path to training data as text')
parser.add_argument('--valid_data_path', type=str, help='Load path to vaidation data as text')
parser.add_argument('--wlist_path', type=str, help='Load path to list of word indices')
parser.add_argument('--unique_prompts_path', type=str, help='Load path to unique prompts as text')
parser.add_argument('--train_prompts_idxs_path', type=str, help='Load path to training data unique prompt indices (for dynamic shuffling)')
parser.add_argument('--valid_prompts_idxs_path', type=str, help='Load path to valid data unique prompt indices (for dynamic shuffling)')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')


# Set device
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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
    prompts_train, prompts_train_lens = text_to_array(args.train_data_path, args.wlist_path)
    prompts_valid, prompts_valid_lens = text_to_array(args.valid_data_path, args.wlist_path)
    responses_train, responses_train_lens = text_to_array(args.train_data_path, args.wlist_path)
    responses_valid, responses_valid_lens = text_to_array(args.valid_data_path, args.wlist_path)
    topics, topics_lens = text_to_array(args.unique_prompts_path, args.wlist_path)

    # For dynamic shuffling to generate the negative samples each epoch, we need to make sure the source
    # and destination prompts are not the same.
    prompts_train_idxs = np.loadtxt(args.train_prompts_idxs_path, dtype=np.float32)
    prompts_valid_idxs = np.loadtxt(args.valid_prompts_idxs_path, dtype=np.float32)

    prompts_train = torch.from_numpy(prompts_train)
    prompts_train = prompts_train.float()
    prompts_train_lens = torch.from_numpy(prompts_train_lens)
    prompts_train_lens = prompts_train_lens.float()

    prompts_valid = torch.from_numpy(prompts_valid)
    prompts_valid = prompts_valid.float()
    prompts_valid_lens = torch.from_numpy(prompts_valid_lens)
    prompts_valid_lens = prompts_valid_lens.float()

    responses_train = torch.from_numpy(responses_train)
    responses_train = responses_train.float()
    responses_train_lens = torch.from_numpy(responses_train_lens)
    responses_train_lens = responses_train_lens.float()

    responses_valid = torch.from_numpy(responses_valid)
    responses_valid = responses_valid.float()
    responses_valid_lens = torch.from_numpy(responses_valid_lens)
    responses_valid_lens = responses_valid_lens.float()

    topics = torch.from_numpy(topics)
    topics = topics.float()
    topics_lens = torch.from_numpy(topics_lens)
    topics_lens = topics_lens.float()

    prompts_train_idxs = torch.from_numpy(prompts_train_idxs)
    prompts_train_idxs = prompts_train_idxs.float()
    prompts_valid_idxs = torch.from_numpy(prompts_valid_idxs)
    prompts_valid_idxs = prompts_valid_idxs.float()

    # Store all training dataset in a single wrapped tensor
    train_ds = TensorDataset(prompts_train, prompts_train_idxs, prompts_train_lens, responses_train, responses_train_lens)
    valid_ds = TensorDataset(prompts_valid, prompts_valid_idxs, prompts_valid_lens, responses_valid, responses_valid_lens)

    # Use DataLoader to handle minibatches easily
    train_dl = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size = args.batch_size, shuffle = False)

    # Construct model
    my_model = SimilarityGridModel(hyperparameters)
    my_model = my_model.float()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(my_model.parameters(), lr = args.learning_rate)

    for epoch in range(args.n_epochs):
        my_model.train()
        total_loss = 0
        counter = 0
        for p, p_id, p_len, r, r_len in train_dl:
            # Forward pass
            y_pred, y_true = my_model.forward(p, p_id, p_len, r, r_len)
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
        for p, p_id, p_len, r, r_len in valid_dl:
            y_pred, y_true = my_model.forward(p, p_id, p_len, r, r_len)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            counter += 1
            print(counter)
        valid_loss = total_loss / counter

        # Report results at end of each epoch
        print("Epoch: ", epoch, " Training Loss: ", trn_loss, " Validation Loss: ", valid_loss)

    # Save the model to a file
    file_path = args.save_path+'sgm_seed'+str(args.seed)+'.pt'
    torch.save(my_model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)