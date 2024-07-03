from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to find similar movies using cosine similarity
def find_similar_movies(input_index, features_matrix, k=10):
    cosine_sim = cosine_similarity([features_matrix[input_index]], features_matrix)
    similar_indices = cosine_sim[0].argsort()[-k-1:-1][::-1]
    return similar_indices

# Evaluating the model using Precision@k with a subset of the test data
def precision_at_k(features_matrix, test_indices, train_indices, k=10, subset_size=100):
    precisions = []
    
    subset_indices = np.random.choice(test_indices, size=min(subset_size, len(test_indices)), replace=False)
    for index in subset_indices:
        true_neighbors = set(train_indices)  # Treat all training indices as potential true neighbors
        recommended_neighbors = set(find_similar_movies(index, features_matrix, k))
        
        true_positives = len(true_neighbors & recommended_neighbors)
        precision = true_positives / k
        precisions.append(precision)
    
    return np.mean(precisions)

def compute_precision(features_matrix, test_indices, train_indices, k=10):
    # Compute Precision@10
    precision_k = precision_at_k(features_matrix, test_indices, train_indices, k)
    print(f'Precision@10: {precision_k:.4f}')