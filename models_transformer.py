#! /usr/bin/env python

import torch
import torchvision.models as models
import math

class SketchyReader(torch.nn.Module):
    def __init__(self, hyperparameters, device):

        super(SketchyReader, self).__init__()

        self.hyps = hyperparameters
        self.device = device

        self.encoder = torch.nn.Embedding(self.hyps['VOCAB_SIZE'], self.hyps['EMBD_DIM'])
        self.pos_encoder = PositionalEncoding(self.hyps['EMBD_DIM'], self.hyps['dropout'])

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.hyps['EMBD_DIM'], nhead=self.hyps['nhead'], 
                                                   dim_feedforward=self.hyps['dim_feedforward'], 
                                                   dropout=self.hyps['dropout'],  activation='gelu')
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.hyps['num_encoder_layers'])     

        self.decoder = torch.nn.Linear(self.hyps['EMBD_DIM'], 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_mask(self, lens):
        max_len = torch.max(lens).item()
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len)) 
        mask = (ids < lens.unsqueeze(1))
        mask = mask.bool()
        return mask

    def forward(self, pr_resp, pr_resp_len, batch_size):

        # Calculate context-dependent representation between prompt and response using several layers of transformer model
        # The positional embedding for each word should have also been added to the input embedding to the tranformer
        # Let batch_size = N, max_pr_resp_len = S, embedding_dimension = E
        # In order to use the transformer model, the batch must be dimension 1 in the tensor (not dimension 0)

        # Cut-off excess words
        max_pr_resp_len = torch.max(pr_resp_len)
        pr_resp = pr_resp[:, 0:max_pr_resp_len]

        pr_mask = self._generate_mask(pr_resp_len)   # Shape = [N,S]
        pr_resp_emb = self.encoder(pr_resp)     # Shape = [N,S,E]
        pr_resp_emb = torch.transpose(pr_resp_emb, 0, 1)    # Shape = [S,N,E]
        src = self.pos_encoder(pr_resp_emb)     # Shape = [S,N,E]
        H = self.transformer_encoder(src, src_key_padding_mask = pr_mask)   # Shape = [S,N,E]
        # Extract first hidden vector
        h1 = torch.squeeze(H[0,:,:])   # Shape = [N,E]
        y = torch.sigmoid(torch.squeeze(self.decoder(h1)))
        return y
        


class PositionalEncoding(torch.nn.Module):
    # Using the method defined in attention-is-all-you-need

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)