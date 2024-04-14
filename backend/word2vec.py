import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import re
from sklearn.preprocessing import normalize


def tokenize(s):
    """
    Tokenizes and lowers case of the string s.
    """
    return re.split(r'\W+', s.lower())


def preprocess_data(file_path):
    """
    Reads the CSV file, preprocesses and tokenizes the descriptions.
    """
    df = pd.read_csv(file_path)
    df['description'] = df['description'].apply(eval)
    df['description'] = df['description'].apply(
        lambda x: ' '.join(x))
    df['tokens'] = df['description'].apply(tokenize)
    return df


def train_word2vec(tokens, vector_size=100, window=5, min_count=1):
    """
    Trains a Word2Vec model on the provided tokens.
    """
    model = Word2Vec(sentences=tokens, vector_size=vector_size,
                     window=window, min_count=min_count, workers=4)
    return model


def generate_embeddings(model, data):
    def get_vector(tokens):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    embeddings = data['tokens'].apply(get_vector)
    embeddings = np.vstack(embeddings)
    embeddings = normalize(embeddings, axis=1)
    return embeddings


if __name__ == "__main__":
    file_path = 'pokematch.csv'
    output_path = 'pokemon_embeddings.csv'
    model_path = 'pokemon_word2vec.model'
    data = preprocess_data(file_path)
    model = train_word2vec(data['tokens'])
    model.save(model_path)
    embeddings = generate_embeddings(model, data)
    embeddings_df = pd.DataFrame(
        embeddings, columns=[f'vec_{i}' for i in range(embeddings.shape[1])])
    embeddings_df['name'] = data['name']

    embeddings_df.to_csv(output_path, index=False)
    print(f'Embeddings have been saved to {output_path}.')
