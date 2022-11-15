import torch
import matplotlib.pyplot as plt

from models.matrix_factorization import MatrixFactorizationModel
from models.neural_collab import NeuralCollaborativeModel
from models.wide_and_deep import WideAndDeepModel
import pytorch_lightning as pl


def create_hook(layer, activations):
    """
    Create a PyTorch hook for a layer that stores the activations.
    """

    def hook(_, __, output):
        activations[layer] = output.detach()

    return hook


def print_activations(model, x):
    """
    Perform a forward pass through the model and print the activations of each layer.
    """

    activations = {}
    model.eval()

    # register a forward hook
    layers = [
        m
        for m in model.modules()
        if not isinstance(m, torch.nn.ReLU) or isinstance(m, torch.nn.Dropout)
    ]
    for l in layers:
        l.register_forward_hook(create_hook(l, activations))

    # perform a forward pass
    model(x)

    # plot the activations
    plt.figure(figsize=(20, 4))
    legends = []

    for i, key in enumerate(activations):
        act = activations[key]
        print(
            "layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%"
            % (i, key, act.mean(), act.std(), (act.abs() > 0.97).float().mean() * 100)
        )
        hy, hx = torch.histogram(act, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({key}")

    plt.legend(legends)
    plt.title("activation distribution")


def load_from_checkpoint(model_name, n_users, n_movies) -> pl.LightningModule:
    """
    Load a trained model from the lastest checkpoint.
    """

    if model_name == "wide-deep":
        feature_dims = torch.tensor((n_users, n_movies))
        hidden_dims = (64, 32)
        embedding_dims = 50

        return WideAndDeepModel.load_from_checkpoint(
            "./checkpoints/di7rt2ln/checkpoints/epoch=19-step=1562500.ckpt",
            feature_dims=feature_dims,
            hidden_dims=hidden_dims,
            embedding_dims=embedding_dims,
            dropout=0.5,
        )
    elif model_name == "neural-collab":
        return NeuralCollaborativeModel.load_from_checkpoint(
            "./checkpoints/2faf3tez/checkpoints/epoch=19-step=1562500.ckpt",
            n_users=n_users,
            n_items=n_movies,
            n_factors=50,
            n_hidden=64,
        )
    elif model_name == "matrix-factorization":
        return MatrixFactorizationModel.load_from_checkpoint(
            "./checkpoints/1o8gw1zp/checkpoints/epoch=19-step=3125000.ckpt",
            n_users=n_users,
            n_items=n_movies,
            n_factors=50,
        )


def initialize_model(model_name, n_users, n_movies) -> pl.LightningModule:
    """
    Initialize a PyTorch Lightning model.
    """

    if model_name == "wide-deep":
        feature_dims = torch.tensor((n_users, n_movies))
        hidden_dims = (64, 32)
        embedding_dims = 50

        return WideAndDeepModel(
            feature_dims=feature_dims,
            hidden_dims=hidden_dims,
            embedding_dims=embedding_dims,
            dropout=0.5,
        )
    elif model_name == "neural-collab":
        return NeuralCollaborativeModel(
            n_users, n_movies, num_factors=50, num_hidden=64
        )
    elif model_name == "matrix-factorization":
        return MatrixFactorizationModel(n_users, n_movies, n_factors=50)

    raise ValueError(f"Model '{model_name}' not supported.")
