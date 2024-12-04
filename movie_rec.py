import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_data():
    movies = pd.read_csv('movies.csv')
    movies['name'] = movies['title'].str.extract(r'^(.*)\s\(\d{4}\)$')  
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')       
    movies['title'] = movies['name']
    encoder = OneHotEncoder()
    genres_encoded = encoder.fit_transform(movies[['genres']]).toarray()
    similarity_matrix = cosine_similarity(genres_encoded)
    return movies, similarity_matrix

movies, similarity_matrix = load_and_preprocess_data()

def recommend_movies(movie_name, k=4):
    if movie_name in movies['title'].values:
        movie_index = movies[movies['title'] == movie_name].index[0]
        similarity_scores = similarity_matrix[movie_index]
        similar_indices = similarity_scores.argsort()[-k-1:-1][::-1]
        recommendations = movies.iloc[similar_indices]['title'].tolist()
        return recommendations
    else:
        return []

st.title('Movie Recommendation System')

movie_name = st.selectbox(
    'Choose a movie you like:',
    options=movies['title'].dropna().sort_values().tolist()
)

if st.button('Recommend'):
    recommendations = recommend_movies(movie_name)
    if recommendations:
        st.write('Recommended Movies:')
        for rec in recommendations:
            st.write(f'- {rec}')
    else:
        st.write('No recommendations found.')

