import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import re

#--------------- Functions ------------------------------
fav_pokemon = {'charizard': '1st', 'eevee': '2nd', 'mew': 'Tied 2nd', 'absol': '4th', 'umbreon': '5th', 'lugia': '6th', 'pikachu': 'Tied 6th', 'gardevoir': '8th', 'rayquaza': '9th', 'lucario': '10th', 'gengar': 'Tied 10th', 'ninetales': 'Tied 10th', 'darkrai': '13rd', 'celebi': '14th', 'zorua': '15th', 'giratina': 'Tied 15th', 'sylveon': '17th', 'raichu': 'Tied 17th', 'squirtle': 'Tied 17th', 'mimikyu': '20th', 'glaceon': 'Tied 20th', 'vulpix': '22nd', 'suicune': 'Tied 22nd', 'ampharos': 'Tied 22nd', 'mewtwo': 'Tied 22nd', 'shaymin': '26th', 'gallade': 'Tied 26th', 'entei': 'Tied 26th', 'cyndaquil': '29th', 'reshiram': 'Tied 29th', 'ditto': 'Tied 29th', 'arcanine': 'Tied 29th', 'garchomp': '33rd', 'bulbasaur': 'Tied 33rd', 'jolteon': '35th', 'charmander': 'Tied 35th', 'haxorus': '37th', 'salamence': 'Tied 37th', 'luxray': 'Tied 37th', 'serperior': 'Tied 37th', 'leafeon': 'Tied 37th', 'piplup': '42nd', 'blaziken': 'Tied 42nd', 'decidueye': 'Tied 42nd', 'jirachi': 'Tied 42nd', 'greninja': 'Tied 42nd', 'diancie': '47th', 'azumarill': '48th', 'xerneas': 'Tied 48th', 'shinx': 'Tied 48th', 'articuno': 'Tied 48th', 'lapras': 'Tied 48th', 'treecko': 'Tied 48th', 'houndoom': 'Tied 48th', 'kyurem': 'Tied 48th', 'riolu': '56th', 'groudon': 'Tied 56th', 'vaporeon': 'Tied 56th', 'snivy': 'Tied 56th', 'mightyena': 'Tied 56th', 'milotic': 'Tied 56th', 'flareon': 'Tied 56th', 'fennekin': 'Tied 56th', 'quilava': '64th', 'lopunny': 'Tied 64th', 'banette': 'Tied 64th', 'hoopa': 'Tied 64th', 'delphox': 'Tied 64th', 'keldeo': 'Tied 64th', 'zoroark': 'Tied 64th', 'rotom': 'Tied 64th', 'sableye': 'Tied 64th', 'mudkip': 'Tied 64th', 'latios': 'Tied 64th', 'starmie': 'Tied 64th', 'scizor': 'Tied 64th', 'mismagius': 'Tied 64th', 'kyogre': 'Tied 64th', 'magikarp': '79th', 'flygon': 'Tied 79th', 'metagross': 'Tied 79th', 'ogerpon': 'Tied 79th', 'meowstic': 'Tied 79th', 'heracross': 'Tied 79th', 'crobat': 'Tied 79th', 'growlithe': 'Tied 79th', 'torterra': 'Tied 79th', 'mawile': 'Tied 79th', 'wartortle': 'Tied 79th', 'dratini': '90th', 'ponyta': 'Tied 90th', 'braixen': 'Tied 90th', 'torchic': 'Tied 90th', 'larvitar': 'Tied 90th', 'haunter': 'Tied 90th', 'pachirisu': 'Tied 90th', 'lunala': 'Tied 90th', 'aegislash': 'Tied 90th', 'dragonite': 'Tied 90th', 'nidoking': 'Tied 90th', 'typhlosion': 'Tied 90th', 'latias': 'Tied 90th', 'furfrou': 'Tied 90th', 'cubone': 'Tied 90th', 'noivern': 'Tied 90th', 'totodile': 'Tied 90th', 'arceus': 'Tied 90th', 'chandelure': 'Tied 90th', 'espeon': 'Tied 90th', 'turtwig': 'Tied 90th', 'tyranitar': 'Tied 90th', 'zekrom': 'Tied 90th', 'deoxys': 'Tied 90th', 'furret': '114th', 'espurr': 'Tied 114th', 'emboar': 'Tied 114th', 'primarina': 'Tied 114th', 'chikorita': 'Tied 114th', 'noibat': 'Tied 114th', 'skiddo': 'Tied 114th', 'whimsicott': 'Tied 114th', 'swampert': 'Tied 114th', 'skitty': 'Tied 114th', 'sewaddle': 'Tied 114th', 'goomy': 'Tied 114th', 'swablu': 'Tied 114th', 'dragonair': 'Tied 114th', 'poochyena': 'Tied 114th', 'froakie': 'Tied 114th', 'hydreigon': 'Tied 114th', 'empoleon': 'Tied 114th', 'mareep': 'Tied 114th', 'roserade': 'Tied 114th', 'dusknoir': 'Tied 114th', 'miltank': 'Tied 114th', 'dunsparce': 'Tied 114th', 'yveltal': 'Tied 114th', 'meloetta': '138th', 'manaphy': 'Tied 138th', 'psyduck': 'Tied 138th', 'lickitung': 'Tied 138th', 'butterfree': 'Tied 138th', 'gyarados': 'Tied 138th', 'rillaboom': 'Tied 138th', 'popplio': 'Tied 138th', 'zygarde': 'Tied 138th', 'oshawott': 'Tied 138th', 'tepig': 'Tied 138th', 'dialga': 'Tied 138th', 'electivire': 'Tied 138th', 'tropius': 'Tied 138th', 'golurk': 'Tied 138th', 'breloom': 'Tied 138th', 'rapidash': 'Tied 138th', 'samurott': 'Tied 138th', 'yanmega': 'Tied 138th', 'victini': 'Tied 138th', 'altaria': 'Tied 138th', 'volcarona': 'Tied 138th', 'weavile': 'Tied 138th', 'aggron': 'Tied 138th', 'blastoise': 'Tied 138th', 'staraptor': 'Tied 138th', 'froslass': 'Tied 138th', 'grookey': 'Tied 138th', 'ursaring': 'Tied 138th', 'onix': '167th', 'ralts': 'Tied 167th', 'virizion': 'Tied 167th', 'grovyle': 'Tied 167th', 'swoobat': 'Tied 167th', 'porygon': 'Tied 167th', 'arbok': 'Tied 167th', 'voltorb': 'Tied 167th', 'muk': 'Tied 167th', 'grimer': 'Tied 167th', 'pecharunt': 'Tied 167th', 'pichu': 'Tied 167th', 'wooper': 'Tied 167th', 'ho-oh': 'Tied 167th', 'moltres': 'Tied 167th', 'noctowl': 'Tied 167th', 'zapdos': 'Tied 167th', 'sceptile': 'Tied 167th', 'infernape': 'Tied 167th', 'skeledirge': 'Tied 167th', 'porygon2': 'Tied 167th', 'necrozma': 'Tied 167th', 'rockruff': 'Tied 167th', 'hawlucha': 'Tied 167th', 'chespin': 'Tied 167th', 'blitzle': 'Tied 167th', 'buizel': 'Tied 167th', 'zangoose': 'Tied 167th', 'swellow': 'Tied 167th', 'garganacl': 'Tied 167th', 'melmetal': 'Tied 167th', 'gholdengo': 'Tied 167th', 'snorlax': 'Tied 167th', 'rowlet': 'Tied 167th', 'togepi': 'Tied 167th', 'litten': 'Tied 167th', 'alakazam': 'Tied 167th', 'drifblim': 'Tied 167th', 'honchkrow': 'Tied 167th', 'bonsly': '206th', 'mesprit': 'Tied 206th', 'jigglypuff': 'Tied 206th', 'palkia': 'Tied 206th', 'golduck': 'Tied 206th', 'gastly': 'Tied 206th', 'houndour': 'Tied 206th', 'jumpluff': 'Tied 206th', 'gligar': 'Tied 206th', 'vanilluxe': 'Tied 206th', 'steelix': 'Tied 206th', 'dracovish': 'Tied 206th', 'abra': 'Tied 206th', 'cinderace': 'Tied 206th', 'emolga': 'Tied 206th', 'maractus': 'Tied 206th', 'minccino': 'Tied 206th', 'zubat': 'Tied 206th', 'meowth': 'Tied 206th', 'mothim': 'Tied 206th', 'sneasel': 'Tied 206th', 'beautifly': 'Tied 206th', 'klefki': 'Tied 206th', 'porygon-z': 'Tied 206th', 'marowak': 'Tied 206th', 'scyther': 'Tied 206th', 'thundurus': 'Tied 206th', 'raikou': 'Tied 206th', 'excadrill': 'Tied 206th', 'kommo-o': 'Tied 206th', 'baxcalibur': 'Tied 206th', 'koraidon': 'Tied 206th', 'miraidon': 'Tied 206th', 'clodsire': 'Tied 206th', 'sprigatito': 'Tied 206th', 'zacian': 'Tied 206th', 'dragapult': 'Tied 206th', 'drampa': 'Tied 206th', 'volcanion': 'Tied 206th', 'honedge': 'Tied 206th', 'tornadus': 'Tied 206th', 'terrakion': 'Tied 206th', 'cobalion': 'Tied 206th', 'cubchoo': 'Tied 206th', 'ferrothorn': 'Tied 206th', 'trubbish': 'Tied 206th', 'darmanitan': 'Tied 206th', 'darumaka': 'Tied 206th', 'gigalith': 'Tied 206th', 'uxie': 'Tied 206th', 'magnezone': 'Tied 206th', 'chimchar': 'Tied 206th', 'armaldo': 'Tied 206th', 'spinda': 'Tied 206th', 'manectric': 'Tied 206th', 'aron': 'Tied 206th', 'shuckle': 'Tied 206th', 'spinarak': 'Tied 206th', 'hypno': 'Tied 206th', 'xatu': 'Tied 206th', 'stakataka': 'Tied 206th', 'charjabug': 'Tied 206th', 'ceruledge': 'Tied 206th', 'bisharp': 'Tied 206th', 'vikavolt': 'Tied 206th', 'corviknight': 'Tied 206th', 'braviary': 'Tied 206th', 'gliscor': 'Tied 206th', 'yamper': 'Tied 206th', 'sandslash': 'Tied 206th', 'drifloon': 'Tied 206th', 'golisopod': 'Tied 206th', 'scorbunny': 'Tied 206th', 'cinccino': 'Tied 206th', 'goodra': 'Tied 206th', 'kubfu': 'Tied 206th', 'solgaleo': 'Tied 206th', 'regice': '283rd', 'nihilego': 'Tied 283rd', 'dewgong': 'Tied 283rd', 'dedenne': 'Tied 283rd', 'combee': 'Tied 283rd', 'venusaur': 'Tied 283rd', 'teddiursa': 'Tied 283rd', 'lombre': 'Tied 283rd', 'phione': 'Tied 283rd', 'ludicolo': 'Tied 283rd', 'pidgeot': 'Tied 283rd', 'zigzagoon': 'Tied 283rd', 'delcatty': 'Tied 283rd', 'roselia': 'Tied 283rd', 'clefable': 'Tied 283rd', 'vanillite': 'Tied 283rd', 'chinchou': 'Tied 283rd', 'magmar': 'Tied 283rd', 'pidgey': 'Tied 283rd', 'girafarig': 'Tied 283rd', 'pidove': 'Tied 283rd', 'regirock': 'Tied 283rd', 'geodude': 'Tied 283rd', 'electrode': 'Tied 283rd', 'garbodor': 'Tied 283rd', 'feraligatr': 'Tied 283rd', 'blacephalon': 'Tied 283rd', 'zebstrika': 'Tied 283rd', 'archeops': 'Tied 283rd', 'tyrantrum': 'Tied 283rd', 'chesnaught': 'Tied 283rd', 'carracosta': 'Tied 283rd', 'quagsire': 'Tied 283rd', 'shedinja': 'Tied 283rd', 'brionne': 'Tied 283rd', 'dustox': 'Tied 283rd', 'magmortar': 'Tied 283rd', 'nidoqueen': 'Tied 283rd', 'floatzel': 'Tied 283rd', 'slaking': 'Tied 283rd', 'swanna': 'Tied 283rd', 'seviper': 'Tied 283rd', 'conkeldurr': 'Tied 283rd', 'aerodactyl': 'Tied 283rd', 'tauros': 'Tied 283rd', 'kingdra': 'Tied 283rd', 'electabuzz': 'Tied 283rd', 'kartana': 'Tied 283rd', 'tinkaton': 'Tied 283rd', 'meganium': 'Tied 283rd', 'houndstone': 'Tied 283rd', 'terapagos': 'Tied 283rd', 'ting-lu': 'Tied 283rd', 'kingambit': 'Tied 283rd', 'dachsbun': 'Tied 283rd', 'ursaluna': 'Tied 283rd', 'glastrier': 'Tied 283rd', 'eternatus': 'Tied 283rd', 'runerigus': 'Tied 283rd', 'zeraora': 'Tied 283rd', 'marshadow': 'Tied 283rd', 'magearna': 'Tied 283rd', 'xurkitree': 'Tied 283rd', 'dhelmise': 'Tied 283rd', 'togedemaru': 'Tied 283rd', 'turtonator': 'Tied 283rd', 'araquanid': 'Tied 283rd', 'pikipek': 'Tied 283rd', 'amaura': 'Tied 283rd', 'helioptile': 'Tied 283rd', 'swirlix': 'Tied 283rd', 'pancham': 'Tied 283rd', 'fletchling': 'Tied 283rd', 'genesect': 'Tied 283rd', 'stunfisk': 'Tied 283rd', 'axew': 'Tied 283rd', 'litwick': 'Tied 283rd', 'joltik': 'Tied 283rd', 'ducklett': 'Tied 283rd', 'reuniclus': 'Tied 283rd', 'archen': 'Tied 283rd', 'tirtouga': 'Tied 283rd', 'heatran': 'Tied 283rd', 'mamoswine': 'Tied 283rd', 'drapion': 'Tied 283rd', 'hippowdon': 'Tied 283rd', 'spiritomb': 'Tied 283rd', 'gastrodon': 'Tied 283rd', 'bastiodon': 'Tied 283rd', 'spheal': 'Tied 283rd', 'minun': 'Tied 283rd', 'lotad': 'Tied 283rd', 'elekid': 'Tied 283rd', 'skarmory': 'Tied 283rd', 'politoed': 'Tied 283rd', 'eiscue': 'Tied 283rd', 'jellicent': 'Tied 283rd', 'lilligant': 'Tied 283rd', 'tatsugiri': 'Tied 283rd', 'lillipup': 'Tied 283rd', 'pawniard': 'Tied 283rd', 'shroomish': 'Tied 283rd', 'leavanny': 'Tied 283rd', 'gible': 'Tied 283rd', 'staryu': 'Tied 283rd', 'clefairy': 'Tied 283rd', 'musharna': 'Tied 283rd', 'togekiss': 'Tied 283rd', 'ivysaur': 'Tied 283rd', 'lycanroc': 'Tied 283rd', 'murkrow': 'Tied 283rd', 'cofagrigus': 'Tied 283rd', 'wimpod': 'Tied 283rd', 'incineroar': 'Tied 283rd', 'kirlia': 'Tied 283rd', 'wooloo': 'Tied 283rd', 'poipole': 'Tied 283rd', 'komala': 'Tied 283rd', 'krookodile': 'Tied 283rd', 'oddish': '402nd', 'solosis': 'Tied 402nd', 'appletun': 'Tied 402nd', 'toxtricity': 'Tied 402nd', 'machamp': 'Tied 402nd', 'mantine': 'Tied 402nd', 'skrelp': 'Tied 402nd', 'passimian': 'Tied 402nd', 'pyroar': 'Tied 402nd', 'misdreavus': 'Tied 402nd', 'cherrim': 'Tied 402nd', 'bidoof': 'Tied 402nd', 'hoppip': 'Tied 402nd', 'stunky': 'Tied 402nd', 'pinsir': 'Tied 402nd', 'pyukumuku': 'Tied 402nd', 'purrloin': 'Tied 402nd', 'wormadam': 'Tied 402nd', 'gogoat': 'Tied 402nd', 'azelf': 'Tied 402nd', 'slurpuff': 'Tied 402nd', 'buneary': 'Tied 402nd', 'ribombee': 'Tied 402nd', 'torkoal': 'Tied 402nd', 'rattata': 'Tied 402nd', 'pidgeotto': 'Tied 402nd', 'enamorus': 'Tied 402nd', 'natu': 'Tied 402nd', 'drowzee': 'Tied 402nd', 'wurmple': 'Tied 402nd', 'slowbro': 'Tied 402nd', 'chansey': 'Tied 402nd', 'wingull': 'Tied 402nd', 'flaaffy': 'Tied 402nd', 'makuhita': 'Tied 402nd', 'sealeo': 'Tied 402nd', 'nidorino': 'Tied 402nd', 'pumpkaboo': 'Tied 402nd', 'sandshrew': 'Tied 402nd', 'primeape': 'Tied 402nd', 'vibrava': 'Tied 402nd', 'beedrill': 'Tied 402nd', 'slowpoke': 'Tied 402nd', 'crustle': 'Tied 402nd', 'doduo': 'Tied 402nd', 'scolipede': 'Tied 402nd', 'swinub': 'Tied 402nd', 'granbull': 'Tied 402nd', 'blissey': 'Tied 402nd', 'nidorina': 'Tied 402nd', 'palpitoad': 'Tied 402nd', 'simisear': 'Tied 402nd', 'tangrowth': 'Tied 402nd', 'carvanha': 'Tied 402nd', 'regigigas': 'Tied 402nd', 'illumise': 'Tied 402nd', 'tentacruel': 'Tied 402nd', 'barboach': 'Tied 402nd', 'liepard': 'Tied 402nd', 'karrablast': 'Tied 402nd', 'servine': 'Tied 402nd', 'armarouge': 'Tied 402nd', 'cryogonal': 'Tied 402nd', 'mandibuzz': 'Tied 402nd', 'amoonguss': 'Tied 402nd', 'registeel': 'Tied 402nd', 'grimmsnarl': 'Tied 402nd', 'regieleki': 'Tied 402nd', 'sinistcha': 'Tied 402nd', 'polteageist': 'Tied 402nd', 'delibird': 'Tied 402nd', 'avalugg': 'Tied 402nd', 'bramblin': 'Tied 402nd', 'corsola': 'Tied 402nd', 'unown': 'Tied 402nd', 'phanpy': 'Tied 402nd', 'fuecoco': 'Tied 402nd', 'guzzlord': 'Tied 402nd', 'wigglytuff': 'Tied 402nd', 'minior': 'Tied 402nd', 'aurorus': 'Tied 402nd', 'dracozolt': 'Tied 402nd', 'azurill': 'Tied 402nd', 'plusle': 'Tied 402nd', 'crawdaunt': 'Tied 402nd', 'hydrapple': 'Tied 402nd', 'klinklang': 'Tied 402nd', 'lanturn': 'Tied 402nd', 'victreebel': 'Tied 402nd', 'cradily': 'Tied 402nd', 'cursola': 'Tied 402nd', 'vivillon': 'Tied 402nd', 'castform': 'Tied 402nd', 'sirfetch’d': 'Tied 402nd', 'poliwrath': 'Tied 402nd', 'simipour': 'Tied 402nd', 'malamar': 'Tied 402nd', 'dewott': 'Tied 402nd', 'florges': 'Tied 402nd', 'tsareena': 'Tied 402nd', 'golem': 'Tied 402nd', 'probopass': 'Tied 402nd', 'caterpie': 'Tied 402nd', 'claydol': 'Tied 402nd', 'naganadel': 'Tied 402nd', 'heliolisk': 'Tied 402nd', 'clawitzer': 'Tied 402nd', 'persian': 'Tied 402nd', 'ninjask': 'Tied 402nd', 'parasect': 'Tied 402nd', 'hitmonlee': 'Tied 402nd', 'landorus': 'Tied 402nd', 'rampardos': 'Tied 402nd', 'croagunk': 'Tied 402nd', 'mienshao': 'Tied 402nd', 'orbeetle': 'Tied 402nd', 'fezandipiti': 'Tied 402nd', 'wo-chien': 'Tied 402nd', 'arctibax': 'Tied 402nd', 'frigibax': 'Tied 402nd', 'annihilape': 'Tied 402nd', 'veluza': 'Tied 402nd', 'cetoddle': 'Tied 402nd', 'orthworm': 'Tied 402nd', 'palafin': 'Tied 402nd', 'rabsca': 'Tied 402nd', 'capsakid': 'Tied 402nd', 'grafaiai': 'Tied 402nd', 'mabosstiff': 'Tied 402nd', 'wattrel': 'Tied 402nd', 'bellibolt': 'Tied 402nd', 'charcadet': 'Tied 402nd', 'dolliv': 'Tied 402nd', 'pawmot': 'Tied 402nd', 'nymble': 'Tied 402nd', 'quaxly': 'Tied 402nd', 'wyrdeer': 'Tied 402nd', 'zarude': 'Tied 402nd', 'zamazenta': 'Tied 402nd', 'dreepy': 'Tied 402nd', 'duraludon': 'Tied 402nd', 'cufant': 'Tied 402nd', 'morpeko': 'Tied 402nd', 'indeedee': 'Tied 402nd', 'snom': 'Tied 402nd', 'falinks': 'Tied 402nd', 'milcery': 'Tied 402nd', 'perrserker': 'Tied 402nd', 'obstagoon': 'Tied 402nd', 'impidimp': 'Tied 402nd', 'clobbopus': 'Tied 402nd', 'centiskorch': 'Tied 402nd', 'toxel': 'Tied 402nd', 'cramorant': 'Tied 402nd', 'applin': 'Tied 402nd', 'coalossal': 'Tied 402nd', 'chewtle': 'Tied 402nd', 'eldegoss': 'Tied 402nd', 'sobble': 'Tied 402nd', 'meltan': 'Tied 402nd', 'cosmog': 'Tied 402nd', 'jangmo-o': 'Tied 402nd', 'stufful': 'Tied 402nd', 'salandit': 'Tied 402nd', 'shiinotic': 'Tied 402nd', 'lurantis': 'Tied 402nd', 'mudbray': 'Tied 402nd', 'mareanie': 'Tied 402nd', 'oricorio': 'Tied 402nd', 'torracat': 'Tied 402nd', 'bergmite': 'Tied 402nd', 'phantump': 'Tied 402nd', 'tyrunt': 'Tied 402nd', 'clauncher': 'Tied 402nd', 'dragalge': 'Tied 402nd', 'barbaracle': 'Tied 402nd', 'inkay': 'Tied 402nd', 'litleo': 'Tied 402nd', 'larvesta': 'Tied 402nd', 'deino': 'Tied 402nd', 'durant': 'Tied 402nd', 'rufflet': 'Tied 402nd', 'golett': 'Tied 402nd', 'druddigon': 'Tied 402nd', 'eelektross': 'Tied 402nd', 'klink': 'Tied 402nd', 'sawsbuck': 'Tied 402nd', 'yamask': 'Tied 402nd', 'sigilyph': 'Tied 402nd', 'scraggy': 'Tied 402nd', 'sandile': 'Tied 402nd', 'venipede': 'Tied 402nd', 'tympole': 'Tied 402nd', 'drilbur': 'Tied 402nd', 'pansear': 'Tied 402nd', 'pansage': 'Tied 402nd', 'rhyperior': 'Tied 402nd', 'snover': 'Tied 402nd', 'carnivine': 'Tied 402nd', 'munchlax': 'Tied 402nd', 'bronzong': 'Tied 402nd', 'cranidos': 'Tied 402nd', 'beldum': 'Tied 402nd', 'shuppet': 'Tied 402nd', 'kecleon': 'Tied 402nd', 'cacturne': 'Tied 402nd', 'trapinch': 'Tied 402nd', 'camerupt': 'Tied 402nd', 'sharpedo': 'Tied 402nd', 'swalot': 'Tied 402nd', 'exploud': 'Tied 402nd', 'nincada': 'Tied 402nd', 'shiftry': 'Tied 402nd', 'nuzleaf': 'Tied 402nd', 'hitmontop': 'Tied 402nd', 'smeargle': 'Tied 402nd', 'ledyba': 'Tied 402nd', 'toucannon': 'Tied 402nd', 'dudunsparce': 'Tied 402nd', 'pheromosa': 'Tied 402nd', 'stoutland': 'Tied 402nd', 'smoliv': 'Tied 402nd', 'sneasler': 'Tied 402nd', 'talonflame': 'Tied 402nd', 'munna': 'Tied 402nd', 'grotle': 'Tied 402nd', 'beartic': 'Tied 402nd', 'gulpin': 'Tied 402nd'}

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

def svd_top_k(df, query, vectorizer, words, docs_normed, data, index_to_word, k = 10):
    """
    vectorizer is a tfidf sklearn vectorizer object, words is the words_compressed matrix, which is svd output transposed.
    docs_normed is first svd output normalized.
    """
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words)).squeeze()
    
    sims = docs_normed.dot(query_vec)
    ranks = np.argsort(-sims)[:k+1]
    
    ranked = []
    for r in ranks:
        name = data.name[r]
        #descs = ". ".join(set(data.description[r][:-1].split(". "))) + "."[:2]
        descs = data.description
        pop = "This Pokémon is not in the top 70 percent of popular Pokémon"
        if name.lower() in fav_pokemon:
            pop = fav_pokemon[name.lower()]
        ranked.append((name, descs, pop))
        
    asort = np.argsort(-query_vec)
    top_traits = []
    for x in asort[:5]:
        dimension_col = np.argsort(-words[:,x].squeeze())[:3]
        top_traits.append([index_to_word[i] for i in dimension_col])
        
    return pd.DataFrame(data=ranked,columns=['name','desc', 'pop']),np.array(top_traits).flatten()

def fav_rank(df, query, vectorizer, words, docs_normed, data, fav_name):
    """
    vectorizer is a tfidf sklearn vectorizer object, words is the words_compressed matrix, which is svd output transposed.
    docs_normed is first svd output normalized.
    """
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(np.dot(query_tfidf, words)).squeeze()
    
    sims = docs_normed.dot(query_vec)
    ranks = np.argsort(-sims)
    
    ranked = []
    rank_i = 1
    for r in ranks:
        name = data.name[r]
        print(name)
        print(fav_name)
        if name == fav_name:
            rank_str = str(rank_i)
            if rank_i % 10 == 1:
                return rank_str + "st"
            elif rank_i % 10 == 2:
                return rank_str + "nd"
            elif rank_i % 10 == 3:
                return rank_str + "rd"
            return rank_str + "th"
        rank_i += 1

    
        
    return -1
    
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
        #descs = ". ".join(set(data.description[r][:-1].split(". "))) + "."[:2]
        descs = data.description
        pop = "This Pokémon is not in the top 70 percent of popular Pokémon"
        if name.lower() in fav_pokemon:
            pop = fav_pokemon[name.lower()]
        ranked.append((name, descs, pop))
    return pd.DataFrame(data=ranked,columns=['name','desc', 'pop'])
