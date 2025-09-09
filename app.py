import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')

# TF-IDF + Cosine Similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommender function
def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# --- Streamlit UI ---
st.markdown(
    """
    <style>
        .title {
            font-size:40px;
            font-weight:bold;
            color:#FF4B4B;
            text-align:center;
        }
        .subtitle {
            font-size:18px;
            color:#BBBBBB;
            text-align:center;
            margin-bottom:30px;
        }
        .recommend-box {
            border: 2px solid #FF4B4B;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            background-color: #1E1E1E;
        }
        .recommend-title {
            font-size:22px;
            font-weight:bold;
            margin-bottom:10px;
            color:#FFFFFF;
        }
        .recommend-movie {
            font-size:18px;
            color:#FFD700;
            margin:5px 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find movies similar to your favorites using Machine Learning</div>', unsafe_allow_html=True)

# Dropdown
movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Recommend Button
if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)

    if recommendations:
        st.markdown('<div class="recommend-box">', unsafe_allow_html=True)
        st.markdown(f'<div class="recommend-title">Recommended movies similar to "{selected_movie}"</div>', unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f'<div class="recommend-movie">🍿 {rec}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("❌ Movie not found in dataset. Please try another.")
