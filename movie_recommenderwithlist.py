import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (make sure movies.csv is in the same folder as this script)
movies = pd.read_csv("movies.csv")

print("✅ Dataset loaded successfully! Total movies:", len(movies))

# Fill missing values with empty string
movies['genres'] = movies['genres'].fillna('')

# Step 1: Convert genres into TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Step 2: Compute cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Step 3: Create a reverse map of movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_movies(title, num_recommendations=5):
    if title not in indices:
        return ["❌ Movie not found in dataset! Please check the spelling and try again."]
    
    # Get index of the movie that matches the title
    idx = indices[title]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar movies (skip the first, since it's the movie itself)
    sim_scores = sim_scores[1:num_recommendations+1]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices].tolist()

def show_movies(sample=20):
    """Show some movie titles to help user know what to type"""
    print("\n🎥 Sample movie titles from dataset:")
    print("-----------------------------------")
    print(movies['title'].sample(sample).tolist())
    print("\n👉 Remember: you need to type the full title exactly as shown (including year).")

# --- Interactive Part ---
while True:
    user_input = input("\n🎬 Enter a movie title, type 'list' to see some movies, or 'exit' to quit: ")

    if user_input.lower() == "exit":
        print("👋 Exiting... Enjoy your movies!")
        break
    elif user_input.lower() == "list":
        show_movies()
        continue

    recommendations = recommend_movies(user_input)
    print("\nRecommended movies similar to '" + user_input + "':")
    for i, movie in enumerate(recommendations, start=1):
        print(f"{i}. {movie}")
