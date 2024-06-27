# import pandas as pd
# import ast
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split


# if __name__ == "__main__":

# #korak 1
#     # Load dataset
#     file_path = 'tmdb_5000_movies.csv'
#     movies_df = pd.read_csv(file_path)

#     def parse_list_column(df, column_name):
#         parsed_list = []
#         for i in range(len(df)):
#             items = ast.literal_eval(df[column_name].iloc[i])
#             parsed_list.append(' '.join([item['name'] for item in items]))
#         return parsed_list

#     # Process genres and keywords
#     movies_df['genres'] = parse_list_column(movies_df, 'genres')
#     movies_df['keywords'] = parse_list_column(movies_df, 'keywords')

#     # Process overview
#     tfidf = TfidfVectorizer(stop_words='english')
#     movies_df['overview'] = movies_df['overview'].fillna('')
#     tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

#     # Normalize numerical features
#     numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
#     scaler = MinMaxScaler()
#     movies_df[numerical_features] = scaler.fit_transform(movies_df[numerical_features])



# #korak2
#     from sklearn.feature_extraction.text import CountVectorizer

#     # Combine genres and keywords
#     movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords']

#     # Vectorize combined features
#     count_vectorizer = CountVectorizer(stop_words='english')
#     count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

#     # Give more weight to the overview feature
#     overview_weight = 3
#     weighted_tfidf_matrix = tfidf_matrix * overview_weight

#     # Combine all features into a single feature matrix
#     import numpy as np
#     features_matrix = np.hstack((count_matrix.toarray(), weighted_tfidf_matrix.toarray(), movies_df[numerical_features].values))



# #korak3
#     train_data, test_data = train_test_split(features_matrix, test_size=0.2, random_state=42)



# #korak4
#     from sklearn.neighbors import NearestNeighbors

#     # Train the KNN model
#     knn = NearestNeighbors(metric='cosine', algorithm='brute')
#     knn.fit(train_data)

#     # Save the model to make predictions later
#     import joblib
#     joblib.dump(knn, 'knn_movie_recommender.pkl')



# #korak5
#     # Load the model
#     knn = joblib.load('knn_movie_recommender.pkl')

#     # Predict the nearest neighbors for each movie in the test set
#     distances, indices = knn.kneighbors(test_data, n_neighbors=10)

#     # Display the results
#     for i, idx in enumerate(indices):
#         print(f"Movie: {movies_df.iloc[i]['title']}")
#         print(f"Overview: {movies_df.iloc[i]['overview']}")
#         print("Recommended Movies:")
#         for neighbor in idx:
#             print(f"  - {movies_df.iloc[neighbor]['title']}")
#             print(f"Overview: {movies_df.iloc[neighbor]['overview']}")
#         break
#------------------------------------------------------------------------------------

# import pandas as pd
# import ast
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import NearestNeighbors
# import joblib
# from gensim.models import Word2Vec, KeyedVectors
# from gensim.test.utils import common_texts

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

# # Function to get sentence embeddings for movie overview
# def get_sentence_embedding(sentence, model):
#     words = sentence.split()
#     word_embeddings = [model[word] for word in words if word in model]
#     if word_embeddings:
#         return np.mean(word_embeddings, axis=0)
#     else:
#         return np.zeros(model.vector_size)

# # Step 4: Process overview
# movies_df['overview'] = movies_df['overview'].fillna('')
# movies_df['embedding'] = movies_df['overview'].apply(lambda x: get_sentence_embedding(x, word_vectors))

# # Convert embeddings to a numpy array
# embedding_matrix = np.vstack(movies_df['embedding'].values)

# # Step 5: Normalize numerical features
# numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
# scaler = MinMaxScaler()
# movies_df[numerical_features] = scaler.fit_transform(movies_df[numerical_features])

# # Step 6: Combine genres and keywords
# from sklearn.feature_extraction.text import CountVectorizer
# movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords']

# # Vectorize combined features
# count_vectorizer = CountVectorizer(stop_words='english')
# count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# # Combine all features into a single feature matrix
# features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

# # Step 7: Split the data
# train_data, test_data = train_test_split(features_matrix, test_size=0.2, random_state=42)

# # Step 8: Train the KNN model
# knn = NearestNeighbors(metric='cosine', algorithm='brute')
# knn.fit(train_data)

# # Save the model to make predictions later
# joblib.dump(knn, 'knn_movie_recommender.pkl')

# # Step 9: Function to find similar movies
# def find_similar_movie(input_movie):
#     if 'title' in input_movie:
#         movie_row = movies_df[movies_df['title'].str.lower() == input_movie['title'].lower()]
#         if not movie_row.empty:
#             idx = movie_row.index[0]
#             distances, indices = knn.kneighbors([features_matrix[idx]], n_neighbors=10)
#             print(f"Movie: {movies_df.iloc[idx]['title']}")
#             #print(f"Overview: {movies_df.iloc[idx]['overview']}")
#             print("Recommended Movies:")
#             for neighbor in indices[0]:
#                 print(f"  - {movies_df.iloc[neighbor]['title']}")
#                 #print(f"Overview: {movies_df.iloc[neighbor]['overview']}")
#             return

#     # If the movie is not in the dataset, create a new entry with given attributes
#     new_movie = {
#         'title': input_movie.get('title', 'Unknown'),
#         'genres': input_movie.get('genres', ''),
#         'keywords': input_movie.get('keywords', ''),
#         'overview': input_movie.get('overview', ''),
#         'popularity': input_movie.get('popularity', 0),
#         'vote_average': input_movie.get('vote_average', 0),
#         'vote_count': input_movie.get('vote_count', 0),
#         'revenue': input_movie.get('revenue', 0)
#     }

#     new_embedding = get_sentence_embedding(new_movie['overview'], word_vectors)
#     new_combined_features = count_vectorizer.transform([new_movie['genres'] + ' ' + new_movie['keywords']])
#     new_numerical_features = scaler.transform([[new_movie['popularity'], new_movie['vote_average'], new_movie['vote_count'], new_movie['revenue']]])

#     new_features = np.hstack((new_combined_features.toarray(), new_embedding, new_numerical_features))

#     distances, indices = knn.kneighbors(new_features, n_neighbors=10)
#     print(f"Movie: {new_movie['title']}")
#     #print(f"Overview: {new_movie['overview']}")
#     print("Recommended Movies:")
#     for neighbor in indices[0]:
#         print(f"  - {movies_df.iloc[neighbor]['title']}")
#         #print(f"Overview: {movies_df.iloc[neighbor]['overview']}")

# # Main function to get user input
# def main():
#     movie_title = input("Enter the movie name: ").strip()
#     movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
#     if not movie_row.empty:
#         input_movie = {'title': movie_title}
#     else:
#         print("Movie not found in the dataset. Please provide the following details:")
#         input_movie = {
#             'title': movie_title,
#             'genres': input("Enter genres (space-separated): ").strip(),
#             'keywords': input("Enter keywords (space-separated): ").strip(),
#             'overview': input("Enter overview: ").strip(),
#             'popularity': float(input("Enter popularity: ") or 0),
#             'vote_average': float(input("Enter vote average: ") or 0),
#             'vote_count': int(input("Enter vote count: ") or 0),
#             'revenue': float(input("Enter revenue: ") or 0)
#         }
    
#     find_similar_movie(input_movie)

# if __name__ == "__main__":
#     main()
#---------------------------------------------------------------------------------

# import pandas as pd
# import ast
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import NearestNeighbors
# import joblib
# from gensim.models import Word2Vec, KeyedVectors
# from gensim.test.utils import common_texts

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

# # Function to get sentence embeddings for movie overview
# def get_sentence_embedding(sentence, model):
#     words = sentence.split()
#     word_embeddings = [model[word] for word in words if word in model]
#     if word_embeddings:
#         return np.mean(word_embeddings, axis=0)
#     else:
#         return np.zeros(model.vector_size)

# # Step 4: Process overview
# movies_df['overview'] = movies_df['overview'].fillna('')
# movies_df['embedding'] = movies_df['overview'].apply(lambda x: get_sentence_embedding(x, word_vectors))

# # Convert embeddings to a numpy array
# embedding_matrix = np.vstack(movies_df['embedding'].values)

# # Step 5: Normalize numerical features
# numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
# scaler = MinMaxScaler()
# movies_df[numerical_features] = scaler.fit_transform(movies_df[numerical_features])

# # Step 6: Combine name, genres, keywords, and overview
# from sklearn.feature_extraction.text import CountVectorizer
# movies_df['combined_features'] = movies_df['title'] + ' ' + movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

# # Vectorize combined features
# count_vectorizer = CountVectorizer(stop_words='english')
# count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# # Combine all features into a single feature matrix
# features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

# # Step 7: Split the data
# train_data, test_data = train_test_split(features_matrix, test_size=0.2, random_state=42)

# # Step 8: Train the KNN model
# knn = NearestNeighbors(metric='cosine', algorithm='brute')
# knn.fit(train_data)

# # Save the model to make predictions later
# joblib.dump(knn, 'knn_movie_recommender.pkl')

# # Step 9: Function to find similar movies
# def find_similar_movie(input_movie):
#     if 'title' in input_movie:
#         movie_row = movies_df[movies_df['title'].str.lower() == input_movie['title'].lower()]
#         if not movie_row.empty:
#             idx = movie_row.index[0]
#             distances, indices = knn.kneighbors([features_matrix[idx]], n_neighbors=10)
#             print(f"Movie: {movies_df.iloc[idx]['title']}")
#             print(f"Overview: {movies_df.iloc[idx]['overview']}")
#             print("Recommended Movies:")
#             for neighbor in indices[0]:
#                 print(f"  - {movies_df.iloc[neighbor]['title']}")
#                 print(f"Overview: {movies_df.iloc[neighbor]['overview']}")
#             return

#     # If the movie is not in the dataset, create a new entry with given attributes
#     new_movie = {
#         'title': input_movie.get('title', 'Unknown'),
#         'genres': input_movie.get('genres', ''),
#         'keywords': input_movie.get('keywords', ''),
#         'overview': input_movie.get('overview', ''),
#         'popularity': input_movie.get('popularity', 0),
#         'vote_average': input_movie.get('vote_average', 0),
#         'vote_count': input_movie.get('vote_count', 0),
#         'revenue': input_movie.get('revenue', 0)
#     }

#     new_embedding = get_sentence_embedding(new_movie['overview'], word_vectors)
#     new_combined_features = count_vectorizer.transform([new_movie['title'] + ' ' + new_movie['genres'] + ' ' + new_movie['keywords'] + ' ' + new_movie['overview']])
#     new_numerical_features = scaler.transform([[new_movie['popularity'], new_movie['vote_average'], new_movie['vote_count'], new_movie['revenue']]])

#     new_features = np.hstack((new_combined_features.toarray(), new_embedding, new_numerical_features))

#     distances, indices = knn.kneighbors(new_features, n_neighbors=10)
#     print(f"Movie: {new_movie['title']}")
#     #print(f"Overview: {new_movie['overview']}")
#     print("Recommended Movies:")
#     for neighbor in indices[0]:
#         print(f"  - {movies_df.iloc[neighbor]['title']}")
#         #print(f"Overview: {movies_df.iloc[neighbor]['overview']}")

# # Main function to get user input
# def main():
#     movie_title = input("Enter the movie name: ").strip()
#     movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
#     if not movie_row.empty:
#         input_movie = {'title': movie_title}
#     else:
#         print("Movie not found in the dataset. Please provide the following details:")
#         input_movie = {
#             'title': movie_title,
#             'genres': input("Enter genres (space-separated): ").strip(),
#             'keywords': input("Enter keywords (space-separated): ").strip(),
#             'overview': input("Enter overview: ").strip(),
#             'popularity': float(input("Enter popularity: ") or 0),
#             'vote_average': float(input("Enter vote average: ") or 0),
#             'vote_count': int(input("Enter vote count: ") or 0),
#             'revenue': float(input("Enter revenue: ") or 0)
#         }
    
#     find_similar_movie(input_movie)

# if __name__ == "__main__":
#     main()
#-----------------------------------------------------

# import pandas as pd
# import ast
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import NearestNeighbors
# import joblib
# from gensim.models import Word2Vec, KeyedVectors
# from gensim.test.utils import common_texts

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
# def get_sentence_embedding(title, overview, model, overview_weight=2.0):
#     title_words = title.split()
#     overview_words = overview.split()
    
#     title_embeddings = [model[word] for word in title_words if word in model]
#     overview_embeddings = [model[word] for word in overview_words if word in model]
    
#     if title_embeddings or overview_embeddings:
#         combined_embeddings = title_embeddings + (overview_embeddings * int(overview_weight))
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
# from sklearn.feature_extraction.text import CountVectorizer
# movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

# # Vectorize combined features
# count_vectorizer = CountVectorizer(stop_words='english')
# count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# # Combine all features into a single feature matrix
# features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

# # Step 7: Split the data
# train_data, test_data = train_test_split(features_matrix, test_size=0.2, random_state=42)

# # Step 8: Train the KNN model
# knn = NearestNeighbors(metric='cosine', algorithm='brute')
# knn.fit(train_data)

# # Save the model to make predictions later
# joblib.dump(knn, 'knn_movie_recommender.pkl')

# # Step 9: Function to find similar movies
# def find_similar_movie(input_movie):
#     if 'title' in input_movie:
#         movie_row = movies_df[movies_df['title'].str.lower() == input_movie['title'].lower()]
#         if not movie_row.empty:
#             idx = movie_row.index[0]
#             distances, indices = knn.kneighbors([features_matrix[idx]], n_neighbors=10)
#             print(f"Movie: {movies_df.iloc[idx]['title']}")
#             print(f"Overview: {movies_df.iloc[idx]['overview']}")
#             print("Recommended Movies:")
#             for neighbor in indices[0]:
#                 print(f"  - {movies_df.iloc[neighbor]['title']}")
#                 print(f"Overview: {movies_df.iloc[neighbor]['overview']}")
#             return

#     # If the movie is not in the dataset, create a new entry with given attributes
#     new_movie = {
#         'title': input_movie.get('title', 'Unknown'),
#         'genres': input_movie.get('genres', ''),
#         'keywords': input_movie.get('keywords', ''),
#         'overview': input_movie.get('overview', ''),
#         'popularity': input_movie.get('popularity', 0),
#         'vote_average': input_movie.get('vote_average', 0),
#         'vote_count': input_movie.get('vote_count', 0),
#         'revenue': input_movie.get('revenue', 0)
#     }

#     new_embedding = get_sentence_embedding(new_movie['title'], new_movie['overview'], word_vectors)
#     new_combined_features = count_vectorizer.transform([new_movie['genres'] + ' ' + new_movie['keywords'] + ' ' + new_movie['overview']])
#     new_numerical_features = scaler.transform([[new_movie['popularity'], new_movie['vote_average'], new_movie['vote_count'], new_movie['revenue']]])

#     new_features = np.hstack((new_combined_features.toarray(), new_embedding, new_numerical_features))

#     distances, indices = knn.kneighbors(new_features, n_neighbors=10)
#     print(f"Movie: {new_movie['title']}")
#     print(f"Overview: {new_movie['overview']}")
#     print("Recommended Movies:")
#     for neighbor in indices[0]:
#         print(f"  - {movies_df.iloc[neighbor]['title']}")
#         print(f"Overview: {movies_df.iloc[neighbor]['overview']}")

# # Main function to get user input
# def main():
#     movie_title = input("Enter the movie name: ").strip()
#     movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
#     if not movie_row.empty:
#         input_movie = {'title': movie_title}
#     else:
#         print("Movie not found in the dataset. Please provide the following details:")
#         input_movie = {
#             'title': movie_title,
#             'genres': input("Enter genres (space-separated): ").strip(),
#             'keywords': input("Enter keywords (space-separated): ").strip(),
#             'overview': input("Enter overview: ").strip(),
#             'popularity': float(input("Enter popularity: ") or 0),
#             'vote_average': float(input("Enter vote average: ") or 0),
#             'vote_count': int(input("Enter vote count: ") or 0),
#             'revenue': float(input("Enter revenue: ") or 0)
#         }
    
#     find_similar_movie(input_movie)

# if __name__ == "__main__":
#     main()
#______________________________________________________\

# import pandas as pd
# import ast
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import NearestNeighbors
# import joblib
# from gensim.models import Word2Vec
# from gensim.models import KeyedVectors
# from gensim.test.utils import common_texts
# from sklearn.metrics import accuracy_score

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
# # Train the Word2Vec model using common texts (you should use your own corpus)
# model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
# model.wv.save("word2vec.wordvectors")

# # Load the word vectors
# word_vectors = KeyedVectors.load("word2vec.wordvectors", mmap='r')

# # Function to get sentence embeddings for movie overview
# def get_sentence_embedding(title, overview, model, overview_weight=2.0):
#     title_words = title.split()
#     overview_words = overview.split()

#     title_embeddings = [model[word] for word in title_words if word in model]
#     overview_embeddings = [model[word] for word in overview_words if word in model]

#     if title_embeddings or overview_embeddings:
#         combined_embeddings = title_embeddings + (overview_embeddings * int(overview_weight))
#         return np.mean(combined_embeddings, axis=0)
#     else:
#         return np.zeros(model.vector_size)

# #Step 4: Process overview
# movies_df['overview'] = movies_df['overview'].fillna('')
# movies_df['title'] = movies_df['title'].fillna('')
# movies_df['embedding'] = movies_df.apply(lambda row: get_sentence_embedding(row['title'], row['overview'], word_vectors), axis=1)

# # Convert embeddings to a numpy array
# embedding_matrix = np.vstack(movies_df['embedding'].values)

# # Step 5: Normalize numerical features
# numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
# scaler = MinMaxScaler()
# movies_df[numerical_features] = scaler.fit_transform(movies_df[numerical_features])

# # Step 6: Combine genres and keywords
# from sklearn.feature_extraction.text import CountVectorizer
# movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords']

# # Vectorize combined features
# count_vectorizer = CountVectorizer(stop_words='english')
# count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# # Combine all features into a single feature matrix
# features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

# # Step 7: Split the data
# train_data, test_data = train_test_split(features_matrix, test_size=0.2, random_state=42)

# # Step 8: Train the KNN model
# knn = NearestNeighbors(metric='cosine', algorithm='brute')
# knn.fit(train_data)

# # Save the model to make predictions later
# joblib.dump(knn, 'knn_movie_recommender.pkl')

# # Step 9: Predict the nearest neighbors for each movie in the test set
# knn = joblib.load('knn_movie_recommender.pkl')
# distances, indices = knn.kneighbors(test_data, n_neighbors=10)

# # Display the results
# for i, idx in enumerate(indices):
#     print(f"Movie: {movies_df.iloc[i]['title']}")
#     print(f"Overview: {movies_df.iloc[i]['overview']}")
#     print("Recommended Movies:")
#     for neighbor in idx:
#         print(f"  - {movies_df.iloc[neighbor]['title']}")
#         print(f"Overview: {movies_df.iloc[neighbor]['overview']}")
#     print("-----------------------------")
#     if i == 1:break


import pandas as pd
import ast
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
from sklearn.metrics import precision_score

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
        combined_embeddings = title_embeddings + overview_embeddings# * overview_weight
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
from sklearn.feature_extraction.text import CountVectorizer
movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['overview']

# Vectorize combined features
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies_df['combined_features'])

# Combine all features into a single feature matrix
features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, movies_df[numerical_features].values))

# Step 7: Split the data
train_data, test_data, train_indices, test_indices = train_test_split(features_matrix, movies_df.index, test_size=0.2, random_state=42)

# Step 8: Train the KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(train_data)

# Save the model to make predictions later
joblib.dump(knn, 'knn_movie_recommender.pkl')

# Function to find similar movies
def find_similar_movies(input_index, k=10):
    distances, indices = knn.kneighbors([features_matrix[input_index]], n_neighbors=k)
    return indices[0]

# Step 9: Evaluate the model using Precision@k
def precision_at_k(test_indices, k=10):
    precisions = []
    
    for index in test_indices:
        true_neighbors = set(train_indices)  # Treat all training indices as potential true neighbors
        recommended_neighbors = set(find_similar_movies(index, k))
        
        true_positives = len(true_neighbors & recommended_neighbors)
        precision = true_positives / k
        precisions.append(precision)
    
    return np.mean(precisions)

# Compute Precision@10
precision_k = precision_at_k(test_indices, k=10)
print(f'Precision@10: {precision_k:.4f}')

# Main function to get user input
def main():
    movie_title = input("Enter the movie name: ").strip()
    movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
    if not movie_row.empty:
        input_movie = {'title': movie_title}
    else:
        print("Movie not found in the dataset. Please provide the following details:")
        input_movie = {
            'title': movie_title,
            'genres': input("Enter genres (space-separated): ").strip(),
            'keywords': input("Enter keywords (space-separated): ").strip(),
            'overview': input("Enter overview: ").strip(),
            'popularity': float(input("Enter popularity: ") or 0),
            'vote_average': float(input("Enter vote average: ") or 0),
            'vote_count': int(input("Enter vote count: ") or 0),
            'revenue': float(input("Enter revenue: ") or 0)
        }
    
    find_similar_movies(input_movie)

if __name__ == "__main__":
    main()