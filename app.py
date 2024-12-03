from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import os

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

# Load datasets
movies_df = pd.read_csv("J:\\MiniProject\\dataset\\movies.csv")
ratings = pd.read_csv("J:\\MiniProject\\dataset\\ratings.csv")

# Clean titles for content-based recommendation
def clean_title(title):
    return re.sub("[^a-zA-Z0-9]", " ", title)

movies_df["clean_title"] = movies_df["title"].apply(clean_title)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies_df["clean_title"])

# Content-Based Search Function
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies_df.iloc[indices].iloc[::-1]
    return results[["title", "genres"]]

# Collaborative Filtering Function
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.1]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies_df, left_index=True, right_on="movieId")[["score", "title", "genres"]]

@app.route('/')
def index():
    return render_template('index.html')

# Route for content-based recommendations
@app.route('/recommend_by_title', methods=['GET'])
def recommend_by_title():
    title = request.args.get('title', '')
    if not title:
        return jsonify({"error": "Title parameter is required"}), 400
    recommendations = search(title)
    return recommendations.to_json(orient="records")

# Route for collaborative filtering recommendations
@app.route('/recommend_similar', methods=['GET'])
def recommend_similar():
    movie_id = int(request.args.get('movie_id', -1))
    if movie_id not in ratings["movieId"].unique():
        return jsonify({"error": "Movie ID not found"}), 404
    recommendations = find_similar_movies(movie_id)
    return recommendations.to_json(orient="records")

if __name__ == '__main__':  
   app.run(debug=True)
