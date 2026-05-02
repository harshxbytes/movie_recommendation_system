from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

app = Flask(__name__)


df = pickle.load(open('df.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
indices = pickle.load(open('indices.pkl', 'rb'))


def recommend(movie):
    movie = movie.strip().lower()

    # DEBUG (remove later if you want)
    print("User input:", movie)

    # handle wrong input using fuzzy matching
    if movie not in indices:
        matches = get_close_matches(movie, indices.keys(), n=1, cutoff=0.6)

        if matches:
            movie = matches[0]
            print("Matched to:", movie)
        else:
            return ["Movie not found"]

    idx = indices[movie]

    # compute cosine similarity
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # get top 5 similar movies
    sim_indices = sim_scores.argsort()[::-1][1:6]

    recommended_movies = df['title'].iloc[sim_indices].tolist()

    return recommended_movies



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_name = request.form.get('movie')

        if not movie_name:
            return render_template('index.html',
                                   recommendations=["Please enter a movie name"])

        recommendations = recommend(movie_name)

        return render_template('index.html',
                               recommendations=recommendations,
                               movie_name=movie_name)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)