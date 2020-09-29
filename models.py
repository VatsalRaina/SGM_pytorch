import torch
import torchvision.models as models

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

        # Perform dynamic shuffling to duplicate the size of the data and get y_true
        # Embed all words with learnable word embeddings for each word in vocabulary (maybe try w2vec)
        # Construct similarity grid (with multiple channels)
        # Pass through resnet 

        # Possibly try passing different 3-channel grids through different resnets
        # i.e. explore the idea behind hierarchical ensembling at every level in a given model

        max_p_len = torch.max(p_len)
        max_r_len = torch.max(r_len)

        p_emb = self.wordEmbd(p)
        r_emb = self.wordEmbd(r)

        r_emb = r_emb.repeat(1, max_p_len, 1)
        r_emb = torch.reshape(r_emb, (batch_size, max_p_len, max_r_len, self.hyps['EMBD_DIM']))
        r_emb = torch.transpose(r_emb, 1, 2)

        p_emb = p_emb.repeat(1, max_r_len, 1)
        p_emb = torch.reshape(p_emb, (batch_size, max_r_len, max_p_len, self.hyps['EMBD_DIM']))
        
        gridCos = self._cosine_dist(r_emb, p_emb, ax=3)
        # Convert to 4D tensor [batch, height, width, channels] - there is currently 1 channel
        gridCos = torch.unsqueeze(gridCos, 3)


        p_emb2 = self.wordEmbd(p)
        r_emb2 = self.wordEmbd(r)

        r_emb2 = r_emb2.repeat(1, max_p_len, 1)
        r_emb2 = torch.reshape(r_emb2, (batch_size, max_p_len, max_r_len, self.hyps['EMBD_DIM']))
        r_emb2 = torch.transpose(r_emb2, 1, 2)

        p_emb2 = p_emb2.repeat(1, max_r_len, 1)
        p_emb2 = torch.reshape(p_emb2, (batch_size, max_r_len, max_p_len, self.hyps['EMBD_DIM']))
        
        gridCos2 = self._cosine_dist(r_emb2, p_emb2, ax=3)
        gridCos2 = torch.unsqueeze(gridCos2, 3)


        p_emb3 = self.wordEmbd(p)
        r_emb3 = self.wordEmbd(r)

        r_emb3 = r_emb3.repeat(1, max_p_len, 1)
        r_emb3 = torch.reshape(r_emb3, (batch_size, max_p_len, max_r_len, self.hyps['EMBD_DIM']))
        r_emb3 = torch.transpose(r_emb3, 1, 2)

        p_emb3 = p_emb3.repeat(1, max_r_len, 1)
        p_emb3 = torch.reshape(p_emb3, (batch_size, max_r_len, max_p_len, self.hyps['EMBD_DIM']))
        
        gridCos3 = self._cosine_dist(r_emb3, p_emb3, ax=3)
        gridCos3 = torch.unsqueeze(gridCos3, 3)

        grid = torch.cat((gridCos, gridCos2, gridCos3), 3)

        # Pass through resnet-18
        y_1000 = self.resnet18(grid)
        y_pred = torch.sigmoid(self.final_layer(y_1000))

        return y_pred
