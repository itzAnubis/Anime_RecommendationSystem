import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

def load_data():
    anime = pd.read_csv("anime.csv", index_col="Unnamed: 0")
    ratings_df = pd.read_csv("rate.csv")
    vectorizer = TfidfVectorizer()
    genre_matrix = vectorizer.fit_transform(anime['genre'])
    return anime, ratings_df, vectorizer, genre_matrix

def generate_recommendations(user_id, df, ratings_df):
    # Filter the ratings DataFrame to get the movies watched by the specified user
    user_movies = ratings_df[ratings_df['user_id'] == user_id]

    # Filter the 'anime.csv' DataFrame to get the names of the movies watched by the user
    watched_movies = df[df['anime_id'].isin(user_movies['anime_id'])]['name']

    # Get the genres of the movies watched by the user
    user_genres = df[df['anime_id'].isin(user_movies['anime_id'])]['genre'].tolist()

    # Create a new DataFrame with the user's movie genres
    user_movie_genres = pd.DataFrame({'genre': user_genres})

    # Combine the user's movie genres with all movie genres
    combined_genres = pd.concat([df['genre'], user_movie_genres['genre']])

    # Perform one-hot encoding on the combined genres DataFrame
    combined_genres_encoded = pd.get_dummies(combined_genres)

    # Compute the cosine similarity between the user's movie genres and all movie genres
    genre_similarity = cosine_similarity(combined_genres_encoded)

    # Get the indices of movies with closest genres
    closest_indices = genre_similarity[-1, :-1].argsort()[::-1][:5]

    # Exclude the movies that the user has already watched from the recommended movies list
    recommended_movies = df.loc[closest_indices]
    recommended_movies = recommended_movies[~recommended_movies['anime_id'].isin(user_movies['anime_id'])]


    st.write("\nRecommended Animes for User_{}:".format(user_id))
    for _, row in recommended_movies.iterrows():
        anime_name = row['name']
        anime_rating = row['rating']  # Assuming the rating column exists in the 'recommended_movies' DataFrame
        st.write("{} (Rating: {})".format(anime_name, anime_rating))


def get_recommendations(anime_id, anime, genre_matrix, N=10):
    item_vector = genre_matrix[anime['anime_id'] == anime_id]
    cosine_similarities = cosine_similarity(item_vector, genre_matrix)
    sorted_similarities = sorted(enumerate(cosine_similarities.squeeze()), key=lambda x: x[1], reverse=True)
    top_n_recommendations = sorted_similarities[1:N + 1]  # Exclude the original item
    return top_n_recommendations


def main():
    st.title("Anime Recommendation System")

    st.image("STARCAPTAIN_Revenge_Eye_Open_pubgmobile_2.gif",width=700)
    # Load data and create genre matrix
    anime, ratings_df, vectorizer, genre_matrix = load_data()

    # Input field for anime title with autocomplete suggestions
    anime_title_prefix = st.text_input("Enter Anime Title:", key="title_input")
    all_anime_titles = list(set(anime['name'].tolist()))

    # Filter suggestions based on user input
    suggestions = [title for title in all_anime_titles if title.lower().startswith(anime_title_prefix.lower())]

    # Display suggestions below the input field
    st.write("Suggestions:", suggestions[:10])  # Show only top 10 suggestions

    # User interaction button
    if anime_title_prefix and st.button("Recommend Similar Anime"):
        if anime_title_prefix not in all_anime_titles:
            st.error("Anime title not found. Please try a different title.")
        else:
            anime_id = anime[anime['name'] == anime_title_prefix]['anime_id'].tolist()[0]
            recommendations = get_recommendations(anime_id, anime, genre_matrix)

            if recommendations:
                data = []
                for item_idx, similarity in recommendations:
                    recommended_item_id = anime.iloc[item_idx]['anime_id']
                    recommended_name = anime.iloc[item_idx]['name']
                    data.append([recommended_item_id, recommended_name, similarity])

                df = pd.DataFrame(data, columns=["Anime ID", "Name", "Similarity"])
                st.table(df)
            else:
                st.write("No recommendations found.")

    user_id = st.text_input("Enter ID")
    generate_recommendations(user_id, anime, ratings_df)

if __name__ == "__main__":
    main()