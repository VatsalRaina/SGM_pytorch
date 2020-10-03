#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve

from utilities import text_to_array
from models import SimilarityGridModel

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=100, help='Specify the test batch size')
parser.add_argument('--prompts_path', type=str, help='Load path to test prompts as text')
parser.add_argument('--resps_path', type=str, help='Load path to test responses as text')
parser.add_argument('--labels_path', type=str, help='Load path to labels')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--wlist_path', type=str, help='Load path to list of word indices')

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/test.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')   

    device = get_default_device()

    # Load up the trained model
    model = torch.load(args.model_path)
    model.eval().to(device)

    prompts, prompts_lens, _ = text_to_array(args.prompts_path, args.wlist_path)
    responses, responses_lens, _ = text_to_array(args.resps_path, args.wlist_path)
    y_true = np.loadtxt(args.labels_path, dtype=np.float)

    #TEMP!!!
    prompts = prompts[:1000]
    prompts_lens = prompts_lens[:1000]
    responses = responses[:1000]
    responses_lens = responses_lens[:1000]
    y_true = y_true[:1000]

    prompts = torch.from_numpy(prompts).long()
    prompts_lens = torch.from_numpy(prompts_lens).long()
    responses = torch.from_numpy(responses).long()
    responses_lens = torch.from_numpy(responses_lens).long()

    ds = TensorDataset(prompts, prompts_lens, responses, responses_lens)
    dl = DataLoader(ds, batch_size = args.batch_size, shuffle = False)

    y_pred_all = []
    for p, p_len, r, r_len in dl:
        p, p_len, r, r_len = p.to(device), p_len.to(device), r.to(device), r_len.to(device)
        batch_size = p.size()[0]
        y_pred = model.forward(p, p_len, r, r_len, batch_size)
        y_pred = y_pred.cpu().detach().numpy().tolist()
        y_pred_all += y_pred
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




