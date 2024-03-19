from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import pandas as pd
import pickle

app = Flask(__name__)

# Load the recommendation model and related data

# Load the model and related data
model = load_model('recommendation_model.h5')

df = pd.read_csv('ratings.csv')
movie_df = pd.read_csv('movies.csv')


user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

df["rating"] = df["rating"].values.astype(np.float32)




# Function to get movie recommendations for a user
def get_movie_recommendations(user_id):
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]

    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched_index = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched_index))

    ratings = model.predict([user_movie_array[:, 0], user_movie_array[:, 1]]).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched_index[x][0]) for x in top_ratings_indices]

    return recommended_movie_ids


# API endpoint for getting movie recommendations for a user
@app.route('/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    recommendations = get_movie_recommendations(user_id)

    # Get movie details for recommended movies
    recommended_movies = movie_df[movie_df["movieId"].isin(recommendations)]
    recommended_movies_data = []
    for row in recommended_movies.itertuples():
        movie_data = {
            'title': row.title,
            'genres': row.genres,
        }
        recommended_movies_data.append(movie_data)

    return jsonify(recommended_movies_data)

if __name__ == '__main__':
    app.run(debug=True)
