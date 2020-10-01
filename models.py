#! /usr/bin/env python

import torch
import torchvision.models as models
import kornia

class SimilarityGridModel(torch.nn.Module):
    def __init__(self, hyperparameters):

        super(SimilarityGridModel, self).__init__()

        self.hyps = hyperparameters

        self.wordEmbd = torch.nn.Embedding(self.hyps['VOCAB_SIZE'], self.hyps['EMBD_DIM'])
        self.wordEmbd2 = torch.nn.Embedding(self.hyps['VOCAB_SIZE'], self.hyps['EMBD_DIM'])
        self.wordEmbd3 = torch.nn.Embedding(self.hyps['VOCAB_SIZE'], self.hyps['EMBD_DIM'])

        self.resnet18 = models.resnet18()
        self.resnet18.train()

        self.final_layer = torch.nn.Linear(1000, 1)


    def _cosine_dist(self, xx, yy, ax):
        cos = torch.nn.CosineSimilarity(dim=ax, eps=1e-6)
        return cos(xx, yy)

    def forward(self, p, p_len, r, r_len, batch_size):
        # Embed all words with learnable word embeddings for each word in vocabulary (maybe try w2vec)
        # Construct similarity grid (with multiple channels)
        # Pass through resnet 

        # Possibly try passing different 3-channel grids through different resnets
        # i.e. explore the idea behind hierarchical ensembling at every level in a given model

        max_p_len = torch.max(p_len)
        max_r_len = torch.max(r_len)

        p_emb = self.wordEmbd(p)
        r_emb = self.wordEmbd(r)

        # Cut-off excess words
        p_emb = p_emb[:, 0:max_p_len, :]
        r_emb = r_emb[:, 0:max_r_len, :]

        print(r_emb.size())
        r_emb = r_emb.repeat(1, max_p_len, 1)
        print(r_emb.size())
        r_emb = torch.reshape(r_emb, (batch_size, max_p_len, max_r_len, self.hyps['EMBD_DIM']))
        r_emb = torch.transpose(r_emb, 1, 2)

        p_emb = p_emb.repeat(1, max_r_len, 1)
        p_emb = torch.reshape(p_emb, (batch_size, max_r_len, max_p_len, self.hyps['EMBD_DIM']))
        
        gridCos = self._cosine_dist(r_emb, p_emb, ax=3)
        # Convert to 4D tensor [batch, height, width, channels] - there is currently 1 channel
        gridCos = torch.unsqueeze(gridCos, 3)


        p_emb2 = self.wordEmbd2(p)
        r_emb2 = self.wordEmbd2(r)

        r_emb2 = r_emb2.repeat(1, max_p_len, 1)
        r_emb2 = torch.reshape(r_emb2, (batch_size, max_p_len, max_r_len, self.hyps['EMBD_DIM']))
        r_emb2 = torch.transpose(r_emb2, 1, 2)

        p_emb2 = p_emb2.repeat(1, max_r_len, 1)
        p_emb2 = torch.reshape(p_emb2, (batch_size, max_r_len, max_p_len, self.hyps['EMBD_DIM']))
        
        gridCos2 = self._cosine_dist(r_emb2, p_emb2, ax=3)
        gridCos2 = torch.unsqueeze(gridCos2, 3)


        p_emb3 = self.wordEmbd3(p)
        r_emb3 = self.wordEmbd3(r)

        r_emb3 = r_emb3.repeat(1, max_p_len, 1)
        r_emb3 = torch.reshape(r_emb3, (batch_size, max_p_len, max_r_len, self.hyps['EMBD_DIM']))
        r_emb3 = torch.transpose(r_emb3, 1, 2)

        p_emb3 = p_emb3.repeat(1, max_r_len, 1)
        p_emb3 = torch.reshape(p_emb3, (batch_size, max_r_len, max_p_len, self.hyps['EMBD_DIM']))
        
        gridCos3 = self._cosine_dist(r_emb3, p_emb3, ax=3)
        gridCos3 = torch.unsqueeze(gridCos3, 3)

        grid = torch.cat((gridCos, gridCos2, gridCos3), 3)

        # Crop and resize the grid
        # For pytorch, the image should be NCHW (it was NHWC in tensorflow)
        grid = torch.transpose(grid, 1,3)
        # So now the dimensions are : [batch_size, num_channels, max_p_len, max_r_len]
        # Create the bounding boxes for cropping
        zero_zero = torch.zeros([batch_size, 2])
        zero_max = torch.cat(torch.unsqueeze(torch.zeros(batch_size)), torch.unsqueeze(r_len, 1), 1)
        max_max = torch.cat((torch.unsqueeze(p_len, 1), torch.unsqueeze(r_len, 1)), 1)
        max_zero = torch.cat((torch.unsqueeze(p_len, 1), torch.unsqueeze(torch.zeros(batch_size))), 1)
        boxes = torch.cat((torch.unsqueeze(zero_zero, 1), torch.unsqueeze(zero_max, 1), torch.unsqueeze(max_max, 1), torch.unsqueeze(max_zero, 1)), 1)
        grid_proc = kornia.crop_and_resize(grid, boxes, [self.hyps['IMG_WIDTH'], self.hyps['IMG_HEIGHT']])

        # Pass through resnet-18
        y_1000 = self.resnet18(grid_proc)
        y_pred = torch.sigmoid(self.final_layer(y_1000))

        return y_pred
