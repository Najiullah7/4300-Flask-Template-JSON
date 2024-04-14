import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import sys


def tokenize(s):
    """
    Tokenizes and lowers case of the string s.
    """
    return re.split(r'\W+', s.lower())


def load_model(model_path):
    """
    Loads a Word2Vec model from the given path.
    """
    return Word2Vec.load(model_path)


def query_to_vector(query, model, vector_size=100):
    """
    Converts a query string into a vector using the Word2Vec model.
    """
    tokens = tokenize(query)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)


def find_most_similar(query_vec, embeddings_df, k=6):
    """
    Finds the k most similar items based on cosine similarity.
    """
    embeddings = embeddings_df.iloc[:, :-1].values  # Exclude the name column
    similarities = cosine_similarity([query_vec], embeddings).flatten()
    indices_most_similar = np.argsort(-similarities)[:k]

    most_similar = embeddings_df.iloc[indices_most_similar]
    return most_similar


def process_query(query):
    model_path = 'pokemon_word2vec.model'
    embeddings_path = 'pokemon_embeddings.csv'
    description_path = 'pokematch.csv'
    k = 6

    # Load the model and the data
    model = load_model(model_path)
    embeddings_df = pd.read_csv(embeddings_path)
    descriptions_df = pd.read_csv(description_path)

    # Get the query vector
    query_vec = query_to_vector(query, model)

    # Find the most similar based on embeddings
    most_similar = find_most_similar(query_vec, embeddings_df, k)

    # Merge the results with descriptions based on the names
    # This assumes 'name' is the common key between the two dataframes
    result_df = pd.merge(most_similar, descriptions_df[[
                         'name', 'description']], on='name', how='left')

    # Format the result to list of lists: [['name', 'description']]
    result_list = result_df[['name', 'description']].values.tolist()

    return result_list


def main():
    if len(sys.argv) > 1:
        query = sys.argv[1]
        results = process_query(query)
        print("Results for query:", query)
        print(results)
    else:
        print("No query provided")


if __name__ == "__main__":
    main()
