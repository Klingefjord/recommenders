import streamlit as st
import pandas as pd
import pytorch_lightning as pl
import torch

from utils import load_from_checkpoint


@st.cache(show_spinner=False)
def load_data():
    return pd.read_csv("./top_1000_movies.csv")


@st.cache(show_spinner=False)
def load_model(model_name):
    return load_from_checkpoint(model_name, n_users=162542, n_movies=209172)


@st.cache
def calculate_embeddings(model, model_name, movie_ids):
    if model_name == "wide-deep":
        # the wide-deep model has one embedding layer for movies and users,
        # delineated by the offsets.
        return model.deep.embeddings(movie_ids + model.wide.offsets[1]).detach()
    else:
        return model.movie_embeddings(movie_ids).detach()


def main():
    st.markdown("# Movie Recommender")
    st.markdown(
        "These models were trained on the MovieLens 25M dataset. The movie embeddings are used to find movies that users like together. \n\nFull training statistics can be found [here](https://wandb.ai/klingefjord/ml-25m/reports/MovieLens-25M--VmlldzoyOTc2Mjcy?accessToken=bii7yizqufs2ihhj8unybxbc51ycq674zgjlqtv1q35frx17742w1jbafnrbmtsa)."
    )
    st.markdown("***")

    # load data
    with st.spinner(text="Loading Data..."):
        df_movies = load_data()

    titles = df_movies["title"].tolist()

    # select movie
    favorites = st.multiselect("Select your favorite movies", titles)
    favorite_indices = [titles.index(x) for x in favorites]

    # select model
    model_name = st.selectbox(
        "Select a model",
        ["matrix-factorization", "neural-collab", "wide-deep"],
    )

    run_model = st.button("Run model")

    if run_model:
        run_model = False

        # load model
        with st.spinner(text="Loading Model..."):
            model = load_model(model_name)

        # get the movie embeddings for the model
        movie_ids = torch.tensor(df_movies["movieId"].tolist())
        movie_embeddings = calculate_embeddings(model, model_name, movie_ids)

        # get the euclidean distances for all movies from the average of the selected movies
        average_embedding = torch.mean(movie_embeddings[favorite_indices], dim=0)
        distances = torch.sum((movie_embeddings - average_embedding) ** 2, dim=1)
        distances[favorite_indices] = float("inf")

        # sort the results and merge with the titles
        results = {title: value for title, value in zip(titles, distances)}
        results = {k: v for (k, v) in sorted(results.items(), key=lambda kv: kv[1])}

        # display the top 10 movies
        st.markdown("## Recommended Movies")

        for i, title in enumerate(list(results.keys())[:10]):
            st.markdown(f"{i + 1}: {title}")


if __name__ == "__main__":
    main()
