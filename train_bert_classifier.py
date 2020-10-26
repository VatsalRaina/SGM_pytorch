#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
parser.add_argument('--n_samples', type=int, default=1, help='Specify the number of negative samples to take')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--num_topics', type=int, default=379, help='Specify the number of unique topics in training')
parser.add_argument('--reverse', type=bool, default=False, help='If true, then concatenate the response onto prompt instead of other way around')
parser.add_argument('--train_resps_path', type=str, help='Load path to training responses as text')
parser.add_argument('--valid_resps_path', type=str, help='Load path to vaidation responses as text')
parser.add_argument('--unique_prompts_path', type=str, help='Load path to unique prompts as text')
parser.add_argument('--unique_prompts_distribution_path', type=str, help='Load path to distribution of unique prompts')
parser.add_argument('--train_prompts_idxs_path', type=str, help='Load path to training data unique prompt indices (for dynamic shuffling)')
parser.add_argument('--valid_prompts_idxs_path', type=str, help='Load path to valid data unique prompt indices (for dynamic shuffling)')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _shuffle(p_id, r, r_msk, topics_dist, NUM_TOPICS, device):
    # Dynamic shuffling in order to generate negative samples
    bs = list(p_id.size())[0]
    y_true_first = np.ones(bs, dtype=int)
    y_true_second = np.zeros(bs, dtype=int)
    y_true = np.concatenate([y_true_first, y_true_second])
    y_true = torch.from_numpy(y_true)
    y_true = y_true.long().to(device)
    new_p_id = np.random.choice(NUM_TOPICS, bs, p=topics_dist)
    for i in range(bs):
        while (new_p_id[i] == p_id[i]):
            new_p_id[i] = np.random.choice(NUM_TOPICS, 1, p=topics_dist)
    new_p_id = torch.from_numpy(new_p_id)
    new_p_id = new_p_id.long().to(device)
    p_id = torch.cat((p_id, new_p_id), 0)
    r = torch.cat((r, r), 0)
    r_msk = torch.cat((r_msk, r_msk), 0)
    return p_id, r, r_msk, y_true, bs

def _get_prompts(p_id, topics, topics_msks):
    p = torch.index_select(topics, 0, p_id)
    p_msk = torch.index_select(topics_msks, 0, p_id)
    return p, p_msk

def _join_pr_resp(p, p_msk, r, r_msk, reverse):
    # Literally concatenate prompt and response without bothering 
    # to put all the padding on the end
    if reverse:
        pr_resp = torch.cat((r, p), 1)
        pr_resp_msk = torch.cat((r_msk, p_msk), 1)
    else:
        pr_resp = torch.cat((p, r), 1)
        pr_resp_msk = torch.cat((p_msk, r_msk), 1)        
    return pr_resp, pr_resp_msk

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    prompts_train_idxs = np.loadtxt(args.train_prompts_idxs_path, dtype=np.int64)
    topics_dist = np.loadtxt(args.unique_prompts_distribution_path, dtype=np.int32)

    # Normalise
    topics_dist = topics_dist / np.linalg.norm(topics_dist, 1)

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    with open(args.unique_prompts_path) as f:
        topics = f.readlines()
    # Remove whitespaces and convert to lowercase
    topics = [x.strip().lower() for x in topics]

    with open(args.train_resps_path) as f:
        responses = f.readlines()
    # Remove whitespaces and convert to lower case
    responses = [x.strip().lower() for x in responses]

    # Tokenize all the prompts and the responses and then map the tokens to their word IDs
    topic_ids = []
    for sent in topics:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        topic_ids.append(encoded_sent)

    resp_ids = []
    for sent in responses:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        resp_ids.append(encoded_sent)
    
    MAX_LEN_topic = max([len(sen) for sen in topic_ids])
    MAX_LEN_resp = max([len(sen) for sen in resp_ids])
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    topic_ids = pad_sequences(topic_ids, maxlen=MAX_LEN_topic, dtype="long", 
                            value=0, truncating="post", padding="post")

    resp_ids = pad_sequences(resp_ids, maxlen=MAX_LEN_resp, dtype="long", 
                            value=0, truncating="post", padding="post")

    # The attention mask simply makes it explicit which tokens are actual words versus which are padding.
    attention_masks_topic = []
    # For each sentence...
    for sent in topic_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks_topic.append(att_mask)
    attention_masks_resp = []
    for sent in resp_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks_resp.append(att_mask)

    # Convert to torch tensors

    prompts_train_idxs = torch.from_numpy(prompts_train_idxs)
    prompts_train_idxs = prompts_train_idxs.long()

    topic_ids = torch.tensor(topic_ids)
    topic_ids = topic_ids.long()
    topic_ids = topic_ids.to(device)

    attention_masks_topic = torch.tensor(attention_masks_topic)
    attention_masks_topic = attention_masks_topic.long()
    attention_masks_topic = attention_masks_topic.to(device)

    resp_ids = torch.tensor(resp_ids)
    resp_ids = resp_ids.long()
    resp_ids = resp_ids.to(device)

    attention_masks_resp = torch.tensor(attention_masks_resp)
    attention_masks_resp = attention_masks_resp.long()
    attention_masks_resp = attention_masks_resp.to(device)

    # Create the DataLoader for our training set.
    print(prompts_train_idxs.size(0))
    print(resp_ids.size(0))
    print(attention_masks_resp.size(0))
    train_data = TensorDataset(prompts_train_idxs, resp_ids, attention_masks_resp)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    )

    loss_values = []

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    for epoch in range(args.n_epochs):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()      
    # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            p_id = batch[0].to(device)
            r = batch[1].to(device)
            r_msk = batch[2].to(device)
            # Perform dynamic shuffling
            p_id, r, r_msk, y_true, batch_size = _shuffle(p_id, r, r_msk, topics_dist, args.num_topics, device)           
            # Get the prompts from the topics
            p, p_msk = _get_prompts(p_id, topic_ids, attention_masks_topic)
            p, p_msk = p.to(device), p_msk.to(device)
            # Concatenate prompts and responses
            pr_resp, pr_resp_msk = _join_pr_resp(p, p_msk, r, r_msk, args.reverse)
            pr_resp, pr_resp_msk = pr_resp.to(device), pr_resp_msk.to(device)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the loss (rather than the model output) because we
            # have provided the `labels`.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(pr_resp, token_type_ids=None, attention_mask=pr_resp_msk, labels=y_true)
            
            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.
            loss = outputs[0]
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)            
        
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

        # NEED TO DO THE VALIDATION CODE NOW - see the rest of the tutorial at
        # https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1

    # Save the model to a file
    file_path = args.save_path+'bert_classifier_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)