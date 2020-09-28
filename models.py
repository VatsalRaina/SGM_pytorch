import torch
import torchvision.models as models

class SimilarityGridModel(torch.nn.Module):
    def __init__(self, hyperparameters):

        super(SimilarityGridModel, self).__init__()

        self.hyperparameters = hyperparameters

        self.resnet18 = models.resnet18()
        self.resnet18.train()

        self.final_layer = torch.nn.Linear(1000, 1)

    def forward(self, p, p_id, p_len, r, r_len):

        # Perform dynamic shuffling to duplicate the size of the data and get y_true
        # Embed all words with learnable word embeddings for each word in vocabulary (maybe try w2vec)
        # Construct similarity grid (with multiple channels)
        # Pass through resnet 

        # Possibly try passing different 3-channel grids through different resnets
        # i.e. explore the idea behind hierarchical ensembling at every level in a given model

        # Pass through resnet-18
        y_1000 = self.resnet18(grid_3)
        y_pred = torch.sigmoid(self.final_layer(y_1000))

        return y_pred, y_true
