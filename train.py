"""Train the model specified in the arguments and log results to wandb."""

import torch
import pytorch_lightning as pl
from models.matrix_factorization import MatrixFactorizationModel
from models.neural_collab import NeuralCollaborativeModel
from models.wide_and_deep import WideAndDeepModel
from loader import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse
import matplotlib.pyplot as plt


def main(
    dataset_name,
    dataset_path,
    model_name,
    epoch,
    learning_rate,
    batch_size,
    weight_decay,
):
    # prepare the dataset
    dataset, train_loader, validation_loader = prepare_data(
        ratings_path=dataset_path, batch_size=batch_size
    )

    # set up wandb logger
    log_config = {
        "dataset": dataset_name,
        "model": model_name,
        "epoch": epoch,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
    }

    wandb_logger = WandbLogger(name=model_name, project=dataset_name, config=log_config)

    # set up the model
    model = initialize_model(model_name, dataset.num_users(), dataset.num_movies())

    # set up the trainer
    trainer = pl.Trainer(
        max_epochs=epoch,
        gpus=1,
        logger=wandb_logger,
    )

    # train the model
    trainer.fit(model, train_loader, validation_loader)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="../data/ml-25m/ratings.csv")
    parser.add_argument("--dataset_name", default="ml-25m")
    parser.add_argument("--model_name", default="neural-collab")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    args = parser.parse_args()

    main(
        args.dataset_name,
        args.dataset_path,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.save_dir,
    )
