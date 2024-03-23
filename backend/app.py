import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import demo

# gets the data
df = pd.read_csv("pokematch.csv")

# gets matrix of terms and docs
term_mat = pd.read_csv('td_mat.csv').values.tolist()

# gets list of good types. idk why the 0 is there, it just adds it.
good_types = pd.read_csv('goodtypes.csv')['0'].tolist()
# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, 'pokematch.json')

# Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
#     pokedex_df = pd.DataFrame(data['pokedex'])

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):

    print("I am in json search")

    s = query
    k = 6

    answer = demo.top_k(s,term_mat,good_types,k,df)[['name','desc']]
    return answer.to_json(orient='records')


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/pokemon")
def pokemon_search():
    text = request.args.get("title")
    print(json_search(text))
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
