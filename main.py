
##KNN


# import pandas as pd
# import ast
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import Word2Vec, KeyedVectors
# from gensim.test.utils import common_texts
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import NearestNeighbors

# # Step 1: Load dataset
# file_path = 'tmdb_5000_movies.csv'
# movies_df = pd.read_csv(file_path)

# # Function to parse list columns like genres and keywords
# def parse_list_column(df, column_name):
#     parsed_list = []
#     for i in range(len(df)):
#         items = ast.literal_eval(df[column_name].iloc[i])
#         parsed_list.append(' '.join([item['name'] for item in items]))
#     return parsed_list

# # Step 2: Process genres and keywords
# movies_df['genres'] = parse_list_column(movies_df, 'genres')
# movies_df['keywords'] = parse_list_column(movies_df, 'keywords')

# # Step 3: Load or train word vectors
# model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
# model.wv.save("word2vec.wordvectors")

# # Load the word vectors
# word_vectors = KeyedVectors.load("word2vec.wordvectors", mmap='r')

# # Function to get sentence embedding with a weight for overview
# def get_sentence_embedding(title, overview, model, overview_weight=2):
#     title_words = title.split()
#     overview_words = overview.split()
    
#     title_embeddings = [model[word] for word in title_words if word in model]
#     overview_embeddings = [model[word] for word in overview_words if word in model]
    
#     if title_embeddings or overview_embeddings:
#         combined_embeddings = title_embeddings + overview_embeddings * overview_weight
#         return np.mean(combined_embeddings, axis=0)
#     else:
#         return np.zeros(model.vector_size)

# # Step 4: Process overview
# movies_df['overview'] = movies_df['overview'].fillna('')
# movies_df['embedding'] = movies_df.apply(lambda row: get_sentence_embedding(row['title'], row['overview'], word_vectors), axis=1)

# # Convert embeddings to a numpy array
# embedding_matrix = np.vstack(movies_df['embedding'].values)

# # Step 5: Normalize numerical features
# numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
# scaler = MinMaxScaler()
# movies_df[numerical_features] = scaler.fit_transform(movies_df[numerical_features])

# # Step 6: Combine genres, keywords, and overview
# movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

# # Vectorize combined features
# count_vectorizer = CountVectorizer(stop_words='english')
# count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# # Combine all features into a single feature matrix
# features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

# print(f'Features matrix shape: {features_matrix.shape}')  # Debug statement

# # Step 7: Split the data
# train_data, test_data, train_indices, test_indices = train_test_split(features_matrix, movies_df.index, test_size=0.2, random_state=42)
# print(f'Train data shape: {train_data.shape}, Test data shape: {test_data.shape}')  # Debug statement

# # Fit Nearest Neighbors model
# knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto')
# knn_model.fit(train_data)

# # Function to find similar movies using KNN
# def find_similar_movies(input_index, k=10):
#     distances, indices = knn_model.kneighbors([features_matrix[input_index]])
#     similar_indices = indices[0][1:]  # Exclude the first item (itself)
#     return similar_indices

# # Step 8: Evaluate the model using Precision@k with a subset of the test data
# def precision_at_k(test_indices, k=10, subset_size=100):
#     precisions = []
    
#     subset_indices = np.random.choice(test_indices, size=min(subset_size, len(test_indices)), replace=False)
#     for index in subset_indices:
#         true_neighbors = set(train_indices)  # Treat all training indices as potential true neighbors
#         recommended_neighbors = set(find_similar_movies(index, k))
        
#         true_positives = len(true_neighbors & recommended_neighbors)
#         precision = true_positives / k
#         precisions.append(precision)
    
#     return np.mean(precisions)

# # Compute Precision@10
# precision_k = precision_at_k(test_indices, k=10)
# print(f'Precision@10: {precision_k:.4f}')

# # Main function to get user input and recommend movies
# def main():
#     movie_title = input("Enter the movie name: ").strip()
#     movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
#     if not movie_row.empty:
#         input_index = movie_row.index[0]
#         similar_movies_indices = find_similar_movies(input_index)
        
#         # Print movie details
#         print(f"Movie details for '{movie_title}':")
#         print("\n")
#         print(f"Title: {movie_row.iloc[0]['title']}")
#         print(f"Overview: {movie_row.iloc[0]['overview']}")
#         print("\n")
        
#         # Print similar movies
#         print(f"Movies similar to '{movie_title}':")
#         print("\n")
#         for idx in similar_movies_indices:
#             print(f"Title: {movies_df.iloc[idx]['title']}")
#             print(f"Overview: {movies_df.iloc[idx]['overview']}")
#             print("\n")


# if __name__ == "__main__":
#     main()



#COSINE SIMILARITY


import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
file_path = 'tmdb_5000_movies.csv'
movies_df = pd.read_csv(file_path)

# Function to parse list columns like genres and keywords
def parse_list_column(df, column_name):
    parsed_list = []
    for i in range(len(df)):
        items = ast.literal_eval(df[column_name].iloc[i])
        parsed_list.append(' '.join([item['name'] for item in items]))
    return parsed_list

# Step 2: Process genres and keywords
movies_df['genres'] = parse_list_column(movies_df, 'genres')
movies_df['keywords'] = parse_list_column(movies_df, 'keywords')

# Step 3: Load or train word vectors
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.wv.save("word2vec.wordvectors")

# Load the word vectors
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

# Step 4: Process overview
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['embedding'] = movies_df.apply(lambda row: get_sentence_embedding(row['title'], row['overview'], word_vectors), axis=1)

# Convert embeddings to a numpy array
embedding_matrix = np.vstack(movies_df['embedding'].values)

# Step 5: Normalize numerical features
numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
scaler = MinMaxScaler()
movies_df[numerical_features] = scaler.fit_transform(movies_df[numerical_features])

# Step 6: Combine genres, keywords, and overview
movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

# Vectorize combined features
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# Combine all features into a single feature matrix
features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

print(f'Features matrix shape: {features_matrix.shape}')  # Debug statement

# Step 7: Split the data
train_data, test_data, train_indices, test_indices = train_test_split(features_matrix, movies_df.index, test_size=0.2, random_state=42)
print(f'Train data shape: {train_data.shape}, Test data shape: {test_data.shape}')  # Debug statement

# Function to find similar movies using cosine similarity
def find_similar_movies(input_index, k=10):
    cosine_sim = cosine_similarity([features_matrix[input_index]], features_matrix)
    similar_indices = cosine_sim[0].argsort()[-k-1:-1][::-1]
    return similar_indices

# Step 8: Evaluate the model using Precision@k with a subset of the test data
def precision_at_k(test_indices, k=10, subset_size=100):
    precisions = []
    
    subset_indices = np.random.choice(test_indices, size=min(subset_size, len(test_indices)), replace=False)
    for index in subset_indices:
        true_neighbors = set(train_indices)  # Treat all training indices as potential true neighbors
        recommended_neighbors = set(find_similar_movies(index, k))
        
        true_positives = len(true_neighbors & recommended_neighbors)
        precision = true_positives / k
        precisions.append(precision)
    
    return np.mean(precisions)

# Compute Precision@10
precision_k = precision_at_k(test_indices, k=10)
print(f'Precision@10: {precision_k:.4f}')

# Main function to get user input and recommend movies
def main():
    movie_title = input("Enter the movie name: ").strip()
    movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
    if not movie_row.empty:
        input_index = movie_row.index[0]
        similar_movies_indices = find_similar_movies(input_index)
        
        # Print movie details
        print(f"Movie details for '{movie_title}':")
        print("\n")
        print(f"Title: {movie_row.iloc[0]['title']}")
        print(f"Overview: {movie_row.iloc[0]['overview']}")
        print("\n")
        
        # Print similar movies
        print(f"Movies similar to '{movie_title}':")
        print("\n")
        for idx in similar_movies_indices:
            print(f"Title: {movies_df.iloc[idx]['title']}")
            print(f"Overview: {movies_df.iloc[idx]['overview']}")
            print("\n")


if __name__ == "__main__":
    main()