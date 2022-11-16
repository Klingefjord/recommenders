# MovieLens 25M

Demo URL: https://klingefjord-recommenders-streamlit-app-ri6m6z.streamlit.app/

This is a cross-comparison of three recommender systems trained on the [MovieLens 25M dataset](https://grouplens.org/datasets/movielens/25m/).

The three models are Vanilla matrix factorization with bias, a neural network with a concatenated embedding vector, and the [Wide and Deep model](https://arxiv.org/abs/1606.07792). 

The models were trained in batches of 256 with a learning rate of `1e-3` for 20 epochs. The full training report can be viewed [here](https://wandb.ai/klingefjord/ml-25m/reports/MovieLens-25M--VmlldzoyOTgwMTUw?accessToken=czyqikd9e32a9f5omtfec8z44le7g3mwa3462c0wyumbxyh3sk1s809vfxqynjga). The neural network models performed far better than matrix factorization. 

<img alt="Validation loss" src="./val_loss_all.png" width="500" />

Wide and deep performed slightly better, but the difference would probably have been bigger if more features were used in training.

<img alt="Comparison between the wide and deep and neural collab models" src="./val_loss.png" width="500" />

