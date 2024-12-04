import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv')

movies['name'] = movies['title'].str.extract(r'^(.*)\s\(\d{4}\)$')  
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')       
movies['title'] = movies['name'] 

encoder = OneHotEncoder()
genres_encoded = encoder.fit_transform(movies[['genres']]).toarray()

movies_features = genres_encoded


def recommend_movies(movie_name, k=4):
    movie_index = movies[movies['title'] == movie_name].index[0]
    
    similarities = cosine_similarity([movies_features[movie_index]], movies_features)
    
    similar_indices = similarities.argsort()[0][-k-1:-1][::-1]
    
    recommendations = movies.iloc[similar_indices]['title'].tolist()
    return recommendations

st.title('Movie Recommendation System')

movie_name = st.text_input('Enter a movie you like:')
if st.button('Recommend'):
    if movie_name in movies['title'].values:
        recommendations = recommend_movies(movie_name)
        st.write('Recommended Movies:')
        for rec in recommendations:
            st.write(f'- {rec}')
    else:
        st.write('Movie not found. Please try another title.')
