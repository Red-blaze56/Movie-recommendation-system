from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess movies
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if title not in movie_indices:
        return jsonify({"error": "Movie not found"}), 404
    idx = movie_indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies
    movie_indices_list = [i[0] for i in sim_scores]
    recommendations = movies['title'].iloc[movie_indices_list].tolist()
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
