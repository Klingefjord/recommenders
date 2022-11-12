import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from loader import prepare_data


class MatrixFactorizationModel(pl.LightningModule):
    """
    A vanilla matrix factorization model with bias.
    """

    def __init__(self, n_users, n_items, n_factors, y_range=(0, 5.5)):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.movie_embeddings = nn.Embedding(n_items, n_factors)

        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_items, 1)

        #
        # initialize embeddings so that the output activations are roughly in the middle of the y_range.
        #
        # tgt = (y_range[1] - y_range[0]) / 2
        # (x / 2)**2*num_factors = tgt
        # (x / 2)**2 = tgt / num_factors
        # x / 2 = math.sqrt(tgt / num_factors)
        # x = math.sqrt(tgt / num_factors) * 2
        #
        tgt = (y_range[1] - y_range[0]) / 2
        self.user_embeddings.weight.data.uniform_(0, math.sqrt(tgt / n_factors) * 2)
        self.movie_embeddings.weight.data.uniform_(0, math.sqrt(tgt / n_factors) * 2)

        # initialize the biases.
        self.user_bias.weight.data.uniform_(-0.5, 0.5)
        self.movie_bias.weight.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        users, movies = x[:, 0], x[:, 1]
        user_embedding = self.user_embeddings(users)
        movie_embedding = self.movie_embeddings(movies)
        bias = (self.user_bias(users) + self.movie_bias(movies)).squeeze()
        dot = (user_embedding * movie_embedding).sum(dim=-1)
        return dot + bias

    def training_step(self, batch, _):
        features, targets = batch
        preds = self(features)
        loss = nn.functional.mse_loss(preds, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        features, targets = batch
        preds = self(features)
        loss = nn.functional.mse_loss(preds, targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-5):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        return optimizer
