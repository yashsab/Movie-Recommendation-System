import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from pathlib import Path

# Load and preprocess data
movielens_data_file_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
movielens_zipped_file = keras.utils.get_file("ml-latest-small.zip", movielens_data_file_url, extract=False)
keras_datasets_path = Path(movielens_zipped_file).parents[0]
movielens_dir = keras_datasets_path / "ml-latest-small"

if not movielens_dir.exists():
    with ZipFile(movielens_zipped_file, "r") as zip:
        zip.extractall(path=keras_datasets_path)

df = pd.read_csv("ratings.csv")

user_ids = df["userId"].unique()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users, num_movies = len(user2user_encoded), len(movie2movie_encoded)
min_rating, max_rating = min(df["rating"]), max(df["rating"])

"""Random Train-Test split"""

df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]

# Define model
embedding_size = 50
user_ips, movie_ips = layers.Input(shape=[1]), layers.Input(shape=[1])
user_embedding = layers.Embedding(num_users, embedding_size)(user_ips)
movie_embedding = layers.Embedding(num_movies, embedding_size)(movie_ips)
user_vect, movie_vect = layers.Flatten()(user_embedding), layers.Flatten()(movie_embedding)
prod = layers.dot([user_vect, movie_vect], axes=1)
dense1 = layers.Dense(150, activation='relu', name='dense1')(prod)
dense2 = layers.Dense(50, activation='relu', name='dense2')(dense1)
dense3 = layers.Dense(1, activation='relu', name='dense3')(dense2)
model = Model([user_ips, movie_ips], dense3)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit([x_train[:,0], x_train[:,1]], y_train, batch_size=64, epochs=10, verbose=1)

# Recommendation
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = set(movie_ids) - set(movies_watched_by_user.movieId.values)
movies_not_watched_index = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched_index))
ratings = model.predict([user_movie_array[:,0],user_movie_array[:,1]]).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched_index[x][0]) for x in top_ratings_indices]

# Print recommendations
print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Top 10 movie recommendations")
print("----" * 8)
recommended_movies = df[df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.movieId, ":", row.movie)
