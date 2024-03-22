import pandas as pd
import numpy as np
import re

# gets the data, this is the data input in the 
df = pd.read_json("pokematch.json")

# gets matrix of terms and docs
term_mat = pd.read_json('td_mat.json').values.tolist()

# gets list of good types. idk why the 0 is there, it just adds it.
good_types = pd.read_json('goodtypes.json')['0'].tolist()

#--------------- Functions ------------------------------

def tokenize(s):
    """
    tokenizes the string s, and makes it lowercase too
    arguments:
    s: string
    
    returns:
    list of tokens
    """
    return re.split(r'\W+', s.lower())


def sims(s,term_mat,good_types):
    """
    gives a list of similarities of the pokemons, in the order of term_mat

    arguments:
    s: string that is being compared to pokemons
    term_mat: matrix of term frequencies, # of pokemons x # of good types
    good_types: list of good_types

    returns:
    list of similarities
    
    *note: this function is called in the next function, top_k
    """
    type_idx = dict(zip(good_types,np.arange(len(good_types))))
    
    tokens = tokenize(s)
    v = np.zeros(len(good_types))
    for token in tokens:
        if token in good_types:
            j = type_idx[token]
            v[j] += 1
    top = np.dot(term_mat,v)
    norm_v = np.linalg.norm(v)
    norm_mat = np.linalg.norm(term_mat, axis=1)
    return top/(norm_v * norm_mat)

def top_k(s,term_mat,good_types,k,data):
    """
    gives top k pokemons related to given string s, in decending order

    arguments:
    s: string that is being compared to pokemons
    term_mat: matrix of term frequencies, # of pokemons x # of good types
    good_types: list of good_types
    k: top k documents to be returned
    data: dataframe with the names and descriptions (see code above for getting this)

    returns:
    list of k tuples. each tuple is (pokemon_name: string, similarity: float i think)
    
    """
    cosines = sims(s, term_mat, good_types)
    ranks = np.argsort(cosines)[-k:][::-1]
    ranked = [(data.name[r],data.description[r]) for r in ranks]
    return pd.DataFrame(data=ranked,columns=['name','desc']).to_json()