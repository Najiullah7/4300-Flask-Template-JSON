import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import demo
import re

# gets the data
df = pd.read_csv("pokematch.csv")

# gets matrix of terms and docs
term_mat = pd.read_csv('td_mat.csv').values.tolist()

# gets list of good types. idk why the 0 is there, it just adds it.
good_types = pd.read_csv('goodtypes.csv')['0'].tolist()

app = Flask(__name__)
CORS(app)

def json_search(query):
    s = query
    k = 6
    answer = demo.top_k(s,term_mat,good_types,k,df)[['name','desc', 'pop']]
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
