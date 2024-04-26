import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import demo
import re
from json import JSONEncoder

# Got this from https://pynative.com/python-serialize-numpy-ndarray-into-json
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
# gets the data
pokedex = pd.read_csv("pokematch.csv")
pokedex.rename(columns = {'description':'documents'},inplace=True)

info = pd.read_csv("pokemon_information.csv")

df =  pd.merge(info, pokedex, on='name', how='outer')
df['documents'] = df.description


vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .8, ngram_range=(1,2))
documents = df.documents.fillna('')


td_matrix = vectorizer.fit_transform(documents)


word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}
feature_names = vectorizer.get_feature_names_out()

docs_compressed, s, words_compressed = svds(td_matrix, k=40)
docs_compressed_normed = normalize(docs_compressed)

words_compressed = words_compressed.transpose()
words_compressed_normed = normalize(words_compressed, axis = 1)

app = Flask(__name__)
CORS(app)

def json_search(query):
    k = 6
    answer = demo.svd_top_k(df, query,vectorizer,words_compressed,docs_compressed_normed,df,index_to_word,k)
    return answer.to_json(orient='records')

def fav_poke_position(query, fav_name):
    rank = demo.fav_rank(df, query,vectorizer,words_compressed,docs_compressed_normed,df,fav_name)
    return jsonify(rank)

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/pokemon")
def pokemon_search():
    text = request.args.get("title")
    return json_search(text)

@app.route('/pokemonRanking', methods=['GET'])
def pokemon_ranking():
    text = request.args.get("title")
    input_term = request.args.get('input')
    return fav_poke_position(text, input_term)

@app.route('/pokemonSuggestions', methods=['GET'])
def pokemon_suggestions():
    print(df['name'])
    search_term = request.args.get('search', '').lower()
    suggestions = [pokemon for pokemon in df['name'] if pokemon.lower().startswith(search_term)]
    return jsonify(suggestions)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
