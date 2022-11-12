import torch
import torch.nn as nn
import pytorch_lightning as pl


class WideModel(nn.Module):
    """
    Class to perform a linear transformation of the features.
    """

    def __init__(self, feature_dims) -> None:
        super().__init__()
        offsets = torch.tensor((0, *torch.cumsum(feature_dims, 0)[:-1]))
        self.register_buffer("offsets", offsets)
        self.embeddings = nn.Embedding(sum(feature_dims), 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, x):
        x = x + self.offsets.unsqueeze(0)
        x = self.embeddings(x).sum(dim=1) + self.bias
        return x


class DeepModel(nn.Module):
    """
    Class that holds a MLP.
    """

    def __init__(
        self,
        feature_dims,
        embedding_dims,
        hidden_dims=(64, 32),
        dropout=0.2,
    ) -> None:
        super().__init__()
        offsets = torch.tensor((0, *torch.cumsum(feature_dims, 0)[:-1]))
        self.register_buffer("offsets", offsets)
        self.embeddings = nn.Embedding(sum(feature_dims), embedding_dims)
        self.mlp = self._create_mlp(embedding_dims, hidden_dims, dropout)
        self._initialize_weights()

    def _create_mlp(self, embedding_dims, hidden_dims, dropout) -> nn.Module:
        layers = list()
        input_dims = embedding_dims * 2
        for dims in hidden_dims:
            layers.append(nn.Linear(input_dims, dims))
            layers.append(nn.BatchNorm1d(dims))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dims = dims
        layers.append(nn.Linear(input_dims, 1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        for layer in self.mlp.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        x = x + self.offsets.unsqueeze(0)
        x = self.embeddings(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)


class WideAndDeepModel(pl.LightningModule):
    """
    A PyTorch implementation of the Wide and Deep model with regression loss.
    https://arxiv.org/abs/1606.07792
    """

    def __init__(
        self,
        feature_dims,
        embedding_dims,
        hidden_dims,
        dropout=0.2,
        scaling_factor=5.5,
    ) -> None:
        super().__init__()
        self.deep = DeepModel(feature_dims, embedding_dims, hidden_dims, dropout)
        self.wide = WideModel(feature_dims)
        self.scaling_factor = scaling_factor
        self.loss = nn.MSELoss()

    def forward(self, x):
        wide = self.wide(x).squeeze()
        deep = self.deep(x).squeeze()
        return torch.sigmoid(wide + deep) * self.scaling_factor

    def training_step(self, batch, _):
        x, targets = batch
        preds = self(x)
        loss = self.loss(preds, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, targets = batch
        preds = self(x)
        loss = self.loss(preds, targets)
        self.log("val_loss", loss)

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-5):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
