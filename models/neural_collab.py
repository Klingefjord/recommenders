import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from loader import prepare_data


class NeuralCollaborativeModel(pl.LightningModule):
    """
    A simple neural network recommender model.
    """

    def __init__(
        self, n_users, n_items, n_factors, n_hidden, dropout=0.5, y_range=(0, 5.5)
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.movie_embeddings = nn.Embedding(n_items, n_factors)
        self.scaling_factor = (y_range[1] - y_range[0]) + y_range[0]
        self.ffnn = nn.Sequential(
            nn.Linear(n_factors * 2, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden // 2, 1),
        )

        for module in self.ffnn.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, x):
        users, movies = x[:, 0], x[:, 1]
        user_embedding = self.user_embeddings(users)
        movie_embedding = self.movie_embeddings(movies)
        output = self.ffnn(torch.cat([user_embedding, movie_embedding], dim=1))
        return torch.sigmoid(output) * self.scaling_factor

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
