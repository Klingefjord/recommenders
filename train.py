"""Train the model specified in the arguments and log results to wandb."""

import pytorch_lightning as pl
from loader import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import argparse

from utils import initialize_model


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
