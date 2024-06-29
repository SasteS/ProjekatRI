import numpy as np
from sklearn.neighbors import NearestNeighbors

knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto')

# Fitting Nearest Neighbors model
def fit_knn_model(train_data):
    knn_model.fit(train_data)

# Function to find similar movies using KNN
def find_similar_movies(features_matrix, input_index, k=10):
    distances, indices = knn_model.kneighbors([features_matrix[input_index]])
    similar_indices = indices[0][1:]  # Exclude the first item (itself)
    return similar_indices

# Evaluating the model using Precision@k with a subset of the test data
def precision_at_k(features_matrix, test_indices, train_indices, k=10, subset_size=100):
    precisions = []
    
    subset_indices = np.random.choice(test_indices, size=min(subset_size, len(test_indices)), replace=False)
    for index in subset_indices:
        true_neighbors = set(train_indices)  # Treat all training indices as potential true neighbors
        recommended_neighbors = set(find_similar_movies(features_matrix, index, k))  # fix call to find_similar_movies
        
        true_positives = len(true_neighbors & recommended_neighbors)
        precision = true_positives / k
        precisions.append(precision)
    
    return np.mean(precisions)

def compute_precision(features_matrix, test_indices, train_indices, k=10):
    # Compute Precision@10
    precision_k = precision_at_k(features_matrix, test_indices, train_indices, k)
    print(f'Precision@10: {precision_k:.4f}')