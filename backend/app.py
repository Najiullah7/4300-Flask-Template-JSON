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
df = pd.read_csv("pokemon_information.csv")

vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .8)
documents = df.description
td_matrix = vectorizer.fit_transform(documents)

word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}
feature_names = vectorizer.get_feature_names_out()

docs_compressed, s, words_compressed = svds(td_matrix, k=40)
docs_compressed_normed = normalize(docs_compressed)

words_compressed = words_compressed.transpose()
words_compressed_normed = normalize(words_compressed, axis = 1)

# gets matrix of terms and docs
term_mat = pd.read_csv('td_mat.csv').values.tolist()

# gets list of good types. idk why the 0 is there, it just adds it.
good_types = pd.read_csv('goodtypes.csv')['0'].tolist()

app = Flask(__name__)
CORS(app)

def json_search(query):
    k = 6
    answer, top_words = demo.svd_top_k(df, query,vectorizer,words_compressed,docs_compressed_normed,df,index_to_word,k)
    return answer.to_json(orient='records')

def top_words_search(query):
    k = 6
    answer, top_words = demo.svd_top_k(df, query,vectorizer,words_compressed,docs_compressed_normed,df,index_to_word,k)
    return json.dumps(top_words, cls=NumpyArrayEncoder)

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

@app.route("/topwords")
def pokemon_list():
    text = request.args.get("title")
    return jsonify(top_words_search(text))

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
