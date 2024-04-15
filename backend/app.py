import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import demo
import re

# gets the data
df = pd.read_csv("pokematch.csv")

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
    answer = demo.svd_top_k(query,vectorizer,words_compressed,docs_compressed_normed,k)[['name','desc', 'pop']]
    return answer.to_json(orient='records')

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/pokemon")
def pokemon_search():
    text = request.args.get("title")
    return json_search(text)


if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
