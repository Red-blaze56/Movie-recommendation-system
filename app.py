from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from dotenv import load_dotenv
import os

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

app = Flask(__name__)

# Load datasets
movies_df = pd.read_csv("J:\\MiniProject\\dataset\\movies.csv")
links_df = pd.read_csv("J:\\MiniProject\\dataset\\links.csv")
ratings = pd.read_csv("J:\\MiniProject\\dataset\\ratings.csv")

# merge dataframes
movies_with_tmdb = pd.merge(movies_df, links_df, on="movieId", how='left')


# getting movie details from api
def get_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Returns a dictionary with movie details
    return None



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
    results = movies_with_tmdb.iloc[indices].iloc[::-1]
    recommendations = []
    for _, row in results.iterrows():
        tmdb_id = row.get("tmdbId")
        movie_details = get_movie_details(tmdb_id) if pd.notna(tmdb_id) else {}
        recommendations.append({
            "title": row["title"],
            "genres": row["genres"],
            "poster": movie_details.get("poster_path"),
            "overview": movie_details.get("overview")
        })
    return recommendations

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
    return rec_percentages.head(10).merge(movies_with_tmdb, left_index=True, right_on="movieId")[["score", "title", "genres", "movieId"]]

@app.route('/')
def home_func():
    return render_template('home.html')

@app.route('/Content')
def content_func():
    return render_template('context.html')

@app.route('/collaborative')
def collab_func():
    return render_template('colab.html')

@app.route('/about')
def about_func():
    return render_template('About.html')

# Route for content-based recommendations
@app.route('/recommend_by_title', methods=['GET'])
def recommend_by_title():
    title = request.args.get('title', '')
    if not title:
        return jsonify({"error": "Title parameter is required"}), 400
    recommendations = search(title)
    return jsonify(recommendations)

# Route for collaborative filtering recommendations
@app.route('/recommend_similar', methods=['GET'])
def recommend_similar():
    movie_id = int(request.args.get('movie_id', -1))
    if movie_id not in ratings["movieId"].unique():
        return jsonify({"error": "Movie ID not found"}), 404
    
    recommendations = find_similar_movies(movie_id)

    recommendations["tmdb_details"] = recommendations["movieId"].apply(lambda x: get_movie_details(links_df.loc[links_df["movieId"] == x, "tmdbId"].values[0]))
    recommendations["poster"] = recommendations["tmdb_details"].apply(lambda x: x.get("poster_path"))
    recommendations["overview"] = recommendations["tmdb_details"].apply(lambda x: x["overview"])

    #changed, see if working, only this line.
    recommendations = recommendations.dropna(subset=["title", "genres"])

    return recommendations[["score", "title", "genres", "poster", "overview"]].to_json(orient="records")

if __name__ == '__main__':  
   app.run(debug=True)