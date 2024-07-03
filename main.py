from preprocessing import *
import knn as knn
import cosine_sim as cosine_sim

def main():
    movie_indices = index_mapping.keys()
    
    for index in movie_indices:
        print(movies_df['title'][index])

    movie_title = input("Enter a movie name you would like\nto find similar movies to: ").strip()
    movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
    if not movie_row.empty:
        model = ""
        while model == "":
            model = input("--------------------------------------------------------------------------------------------------------------------------------------------------------\nChoose a model (Knn/cosine): ").strip()

        input_index = movie_row.index[0]
        if input_index in index_mapping.keys():
            if model.lower() == "knn":
                knn.fit_knn_model(train_data)
                similar_movies_indices = knn.find_similar_movies(test_data, index_mapping[input_index])
                knn.compute_precision(features_matrix, test_indices, train_indices)
            elif model.lower() == "cosine":
                similar_movies_indices = cosine_sim.find_similar_movies(index_mapping[input_index], test_data)
                cosine_sim.compute_precision(features_matrix, test_indices, train_indices)
            else:
                print(f"Model '{model}' not recognized. Please choose 'Knn' or 'cosine'.")
                return
        else:
            print(f"Moive '{movie_title}' is not the test_data.")
            return

        # Print movie details
        print("________________________________________________________________________________________________________________________________________________________")
        print(f"Movie details for '{movie_title}':\n")
        print(f"Title: {movie_row.iloc[0]['title']}")
        print(f"Overview: {movie_row.iloc[0]['overview']}\n")
        
        # Print similar movies
        print(f"Movies similar to '{movie_title}':")
        print("========================================================================================================================================================")
        for idx in similar_movies_indices:
            print(f"Title: {movies_df.iloc[idx]['title']}")
            print(f"Overview: {movies_df.iloc[idx]['overview']}\n")
            print("--------------------------------------------------------------------------------------------------------------------------------------------------------")
    else:
        print("________________________________________________________________________________________________________________________________________________________")
        print(f"Movie '{movie_title}' not found in the dataset.")

if __name__ == "__main__":
    while True:
        main()
        print("\n"*5)