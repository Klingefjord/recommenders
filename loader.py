import torch
import typing
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

torch.manual_seed(42)


class MovielensDataset(Dataset):
    """
    The Movielens dataset.
    """

    def __init__(self, df_ratings):
        self.df_ratings = df_ratings

    def __len__(self) -> int:
        return len(self.df_ratings)

    def __getitem__(self, idx) -> typing.Tuple[torch.LongTensor, torch.FloatTensor]:
        # Get the user id and movie id.
        user_id = self.df_ratings.iloc[idx, 0]
        movie_id = self.df_ratings.iloc[idx, 1]
        batch = torch.tensor((user_id, movie_id), dtype=torch.long)

        # Get the rating, 1-5 with 0.5 increments.
        targets = torch.tensor(self.df_ratings.iloc[idx, 2], dtype=torch.float)

        return batch, targets

    def num_users(self) -> int:
        return self.df_ratings["userId"].max() + 1

    def num_movies(self) -> int:
        return self.df_ratings["movieId"].max() + 1


def prepare_data(
    ratings_path="../../data/ml-latest-small/ratings.csv", batch_size=64
) -> typing.Tuple[MovielensDataset, DataLoader, DataLoader]:
    """
    Prepare the data for training and validation.
    """

    df_ratings = pd.read_csv(ratings_path)
    dataset = MovielensDataset(df_ratings)

    split = int(len(dataset) * 0.8)
    train_data, valid_data = random_split(
        dataset,
        [split, len(dataset) - split],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    validation_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, drop_last=True
    )

    return dataset, train_loader, validation_loader
