#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve
import random
import time
import datetime

from transformers import XLNetTokenizer
from transformers import XLNetForSequenceClassification
from keras.preprocessing.sequence import pad_sequences


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--resps_path', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--reverse', type=bool, default=False, help='If true, then concatenate the response onto prompt instead of other way around')


# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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
    with open('CMDs/test.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')   

    device = get_default_device()

    # Load up the trained model
    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    y_true = np.loadtxt(args.labels_path, dtype=np.float)

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    with open(args.prompts_path) as f:
        prompts = f.readlines()
    # Remove whitespaces and convert to lower case
    prompts = [x.strip().lower() for x in prompts]

    with open(args.resps_path) as f:
        responses = f.readlines()
    # Remove whitespaces and convert to lower case
    responses = [x.strip().lower() for x in responses]

    # Tokenize all the prompts and the responses and then map the tokens to their word IDs
    prompt_ids = []
    for sent in prompts:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        prompt_ids.append(encoded_sent)

    resp_ids = []
    for sent in responses:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        resp_ids.append(encoded_sent)
    
    MAX_LEN_prompt = max([len(sen) for sen in prompt_ids])
    MAX_LEN_resp = max([len(sen) for sen in resp_ids])
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    prompt_ids = pad_sequences(prompt_ids, maxlen=MAX_LEN_prompt, dtype="long", 
                            value=0, truncating="post", padding="post")

    resp_ids = pad_sequences(resp_ids, maxlen=MAX_LEN_resp, dtype="long", 
                            value=0, truncating="post", padding="post")

    # The attention mask simply makes it explicit which tokens are actual words versus which are padding.
    attention_masks_prompt = []
    # For each sentence...
    for sent in prompt_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks_prompt.append(att_mask)
    attention_masks_resp = []
    for sent in resp_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks_resp.append(att_mask)

    prompt_ids = torch.tensor(prompt_ids)
    prompt_ids = prompt_ids.long()
    prompt_ids = prompt_ids.to(device)

    attention_masks_prompt = torch.tensor(attention_masks_prompt)
    attention_masks_prompt = attention_masks_prompt.long()
    attention_masks_prompt = attention_masks_prompt.to(device)

    resp_ids = torch.tensor(resp_ids)
    resp_ids = resp_ids.long()
    resp_ids = resp_ids.to(device)

    attention_masks_resp = torch.tensor(attention_masks_resp)
    attention_masks_resp = attention_masks_resp.long()
    attention_masks_resp = attention_masks_resp.to(device)

    y_true_torch = torch.tensor(y_true)
    y_true_torch = y_true_torch.long()
    y_true_torch = y_true_torch.to(device)

    ds = TensorDataset(prompt_ids, attention_masks_prompt, resp_ids, attention_masks_resp, y_true_torch)
    dl = DataLoader(ds, batch_size = args.batch_size, shuffle = False)

    y_pred_all = []
    count = 0
    print(len(dl))
    for p, p_msk, r, r_msk, y_t in dl:
        print(count)
        count+=1
        p, p_msk, r, r_msk, y_t = p.to(device), p_msk.to(device), r.to(device), r_msk.to(device), y_t.to(device)
        # Concatenate prompts and responses
        pr_resp, pr_resp_msk = _join_pr_resp(p, p_msk, r, r_msk, args.reverse)
        pr_resp, pr_resp_msk = pr_resp.to(device), pr_resp_msk.to(device)        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            outputs = model(pr_resp, token_type_ids=None, attention_mask=pr_resp_msk, labels=y_t)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        logits = np.squeeze(logits[:, 1])
        logits = logits.tolist()
        y_pred_all += logits
        # if count==2:
        #     break
    y_pred_all = np.array(y_pred_all)

    # Save the predicted values so that they can be used for ensembling
    np.savetxt(args.predictions_save_path, y_pred_all)

    # Calculate and report best F0.5 score
    # Label 1 indicates on-topic and 0 indicates off-topic
    # But we care about detecting off-topic data, so to use F0.5, we do 1. - all values
    y_true = 1.-y_true
    y_pred = 1.-y_pred_all
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    f_score = np.amax( (1.+0.5**2) * ( (precision * recall) / (0.5**2 * precision + recall) ) )
    print("F0.5 score is:")
    print(f_score)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args) 