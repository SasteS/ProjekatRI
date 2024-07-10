import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
from sklearn.model_selection import train_test_split

# Loading dataset
file_path = 'tmdb_5000_movies.csv'
movies_df = pd.read_csv(file_path)

# Function to parse list columns like genres and keywords
def parse_list_column(df, column_name):
    parsed_list = []
    for i in range(len(df)):
        items = ast.literal_eval(df[column_name].iloc[i])
        parsed_list.append(' '.join([item['name'] for item in items]))
    return parsed_list

# Processing genres and keywords
movies_df['genres'] = parse_list_column(movies_df, 'genres')
movies_df['keywords'] = parse_list_column(movies_df, 'keywords')

# Load/train word vectors
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.wv.save("word2vec.wordvectors")

# Loading word vectors
word_vectors = KeyedVectors.load("word2vec.wordvectors", mmap='r')

# Function to get sentence embedding with a weight for overview
def get_sentence_embedding(title, overview, model, overview_weight=2):
    title_words = title.split()
    overview_words = overview.split()
    
    title_embeddings = [model[word] for word in title_words if word in model]
    overview_embeddings = [model[word] for word in overview_words if word in model]
    
    if title_embeddings or overview_embeddings:
        combined_embeddings = title_embeddings + overview_embeddings * overview_weight
        return np.mean(combined_embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

# Processing overview
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['embedding'] = movies_df.apply(lambda row: get_sentence_embedding(row['title'], row['overview'], word_vectors), axis=1)

# Converting embeddings to a numpy array
embedding_matrix = np.vstack(movies_df['embedding'].values)

# Normalizing numerical features
numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
scaler = MinMaxScaler()
movies_df[numerical_features] = scaler.fit_transform(movies_df[numerical_features])

# Combining genres, keywords, and overview
movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

# Vectorizing combined features
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])


features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

# Splitting the data
train_data, test_data, train_indices, test_indices = train_test_split(features_matrix, movies_df.index, test_size=0.2, random_state=42)

test_movies=[]
with open("test_data.txt", "w", encoding="utf-8") as file:
    for index in test_indices:
        movie = movies_df.iloc[index]
        test_movies.append(movie['title'])
        file.write(f"Title: {movie['title']}\n")
        file.write(f"Overview: {movie['overview']}\n")
        file.write(f"Genres: {movie['genres']}\n")
        file.write(f"Keywords: {movie['keywords']}\n")
        file.write(f"Popularity: {movie['popularity']}\n")
        file.write(f"Vote Average: {movie['vote_average']}\n")
        file.write(f"Vote Count: {movie['vote_count']}\n")
        file.write(f"Revenue: {movie['revenue']}\n")
        file.write("="*80 + "\n")

with open("train_data.txt", "w", encoding="utf-8") as file:
    for index in test_indices:
        movie = movies_df.iloc[index]
        file.write(f"Title: {movie['title']}\n")
        file.write(f"Overview: {movie['overview']}\n")
        file.write(f"Genres: {movie['genres']}\n")
        file.write(f"Keywords: {movie['keywords']}\n")
        file.write(f"Popularity: {movie['popularity']}\n")
        file.write(f"Vote Average: {movie['vote_average']}\n")
        file.write(f"Vote Count: {movie['vote_count']}\n")
        file.write(f"Revenue: {movie['revenue']}\n")
        file.write("="*80 + "\n")
        

