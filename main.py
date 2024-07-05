from preprocessing import *
import knn as knn
import cosine_sim as cosine_sim

def main():
    movie_title = input("Enter a movie name you would like\nto find similar movies to: ").strip()
    movie_row = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    
    if not movie_row.empty:
        model = ""
        while model == "":
            model = input("--------------------------------------------------------------------------------------------------------------------------------------------------------\nChoose a model (Knn/cosine): ").strip()

        input_index = movie_row.index[0]
        movie_found = movies_df.iloc[input_index]
        if movie_found['title'] in test_movies:
            if model.lower() == "knn":
                knn.fit_knn_model(train_data)
                similar_movies_indices = knn.find_similar_movies(features_matrix, input_index)
                knn.compute_precision(features_matrix, test_indices, train_indices)
            elif model.lower() == "cosine":
                similar_movies_indices = cosine_sim.find_similar_movies(input_index, features_matrix)
                cosine_sim.compute_precision(features_matrix, test_indices, train_indices)
            else:
                print(f"Model '{model}' not recognized. Please choose 'Knn' or 'cosine'.")
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
            print(f"Movie '{movie_title}' not found in the test set.")
    else:
        print("________________________________________________________________________________________________________________________________________________________")
        print(f"Movie '{movie_title}' not found in the dataset.")


if __name__ == "__main__":
    while True:
        main()
        print("\n"*5)