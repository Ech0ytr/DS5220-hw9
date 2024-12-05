from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st

movies = pd.read_csv('movies.csv')
movies['name'] = movies['title'].str.extract(r'^(.*)\s\(\d{4}\)$')
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
movies['title'] = movies['name']

encoder = OneHotEncoder()
genres_encoded = csr_matrix(encoder.fit_transform(movies[['genres']]).toarray())

def recommend_movies(movie_name, k=4):
    if movie_name not in movies['title'].values:
        return []
    movie_index = movies[movies['title'] == movie_name].index[0]
    similarity_scores = cosine_similarity(genres_encoded[movie_index], genres_encoded).flatten()
    similar_indices = similarity_scores.argsort()[-k-1:-1][::-1]
    recommendations = movies.iloc[similar_indices]['title'].tolist()
    return recommendations

st.title('Movie Recommendation System')
movie_name = st.text_input('Enter a movie you like:')
if st.button('Recommend'):
    if movie_name in movies['title'].values:
        recommendations = recommend_movies(movie_name)
        if recommendations:
            st.write('Recommended Movies:')
            for rec in recommendations:
                st.write(f'- {rec}')
        else:
            st.write('No recommendations found.')
    else:
        st.write('Movie not found. Please try another title.')

