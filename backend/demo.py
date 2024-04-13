import pandas as pd
import numpy as np
import re

#--------------- Functions ------------------------------
fav_pokemon = {'charizard': '1st', 'eevee': '2nd', 'mew': 'Tied 2nd', 'absol': '4th', 'umbreon': '5th', 'lugia': '6th', 'pikachu': 'Tied 6th', 'gardevoir': '8th', 'rayquaza': '9th', 'lucario': '10th', 'gengar': 'Tied 10th', 'ninetales': 'Tied 10th', 'darkrai': '13rd', 'celebi': '14th', 'zorua': '15th', 'giratina': 'Tied 15th', 'sylveon': '17th', 'raichu': 'Tied 17th', 'squirtle': 'Tied 17th', 'mimikyu': '20th', 'glaceon': 'Tied 20th', 'vulpix': '22nd', 'suicune': 'Tied 22nd', 'ampharos': 'Tied 22nd', 'mewtwo': 'Tied 22nd', 'shaymin': '26th', 'gallade': 'Tied 26th', 'entei': 'Tied 26th', 'cyndaquil': '29th', 'reshiram': 'Tied 29th', 'ditto': 'Tied 29th', 'arcanine': 'Tied 29th', 'garchomp': '33rd', 'bulbasaur': 'Tied 33rd', 'jolteon': '35th', 'charmander': 'Tied 35th', 'haxorus': '37th', 'salamence': 'Tied 37th', 'luxray': 'Tied 37th', 'serperior': 'Tied 37th', 'leafeon': 'Tied 37th', 'piplup': '42nd', 'blaziken': 'Tied 42nd', 'decidueye': 'Tied 42nd', 'jirachi': 'Tied 42nd', 'greninja': 'Tied 42nd', 'diancie': '47th', 'azumarill': '48th', 'xerneas': 'Tied 48th', 'shinx': 'Tied 48th', 'articuno': 'Tied 48th', 'lapras': 'Tied 48th', 'treecko': 'Tied 48th', 'houndoom': 'Tied 48th', 'kyurem': 'Tied 48th'}
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
    list of k tuples. each tuple is (pokemon_name: string, desc: description)
    
    """
    cosines = sims(s, term_mat, good_types)
    ranks = np.argsort(cosines)[-k:][::-1]
    ranked = []
    for r in ranks:
        name = data.name[r]
        descs = ". ".join(set(data.description[r][:-1].split(". "))) + "."[:2]
        pop = -1
        if name.lower() in fav_pokemon:
            pop = fav_pokemon[name.lower()]
        ranked.append((name, descs, pop))
    return pd.DataFrame(data=ranked,columns=['name','desc', 'pop'])
