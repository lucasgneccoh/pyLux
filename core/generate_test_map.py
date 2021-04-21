# Create the elements of the classic world map

countries = [
  # South America
  {'id': 'VEN', 'name':'Venezuela', 'xy':'250,380', 'card_kind':2},
  {'id': 'BRA', 'name':'Brazil', 'xy':'400,440', 'card_kind':1},
  {'id': 'PER', 'name':'Peru', 'xy':'250,580', 'card_kind':2},

  #North America
  {'id': 'MEX', 'name':'Mexico', 'xy':'110,250', 'card_kind':1},
  {'id': 'GRE', 'name':'Greenland - Nunavut', 'xy':'300, 10', 'card_kind':3},

  # Europe
  {'id': 'ICE', 'name':'Iceland', 'xy':'480,80', 'card_kind':3},
  {'id': 'SEU', 'name':'Southern Europe', 'xy':'600,200', 'card_kind':2},

  # Africa
  {'id': 'NAF', 'name':'North Africa', 'xy':'640,350', 'card_kind':2},
  {'id': 'EGY', 'name':'Egypt', 'xy':'810,320', 'card_kind':1}
  ]
continents = [
  {'name':'South America', 'bonus':2, 'countries':['VEN','BRA','PER']},
  {'name':'North America', 'bonus':5, 'countries':['MEX', 'GRE']},
  {'name':'Europe', 'bonus':5, 'countries':['ICE', 'SEU']},
  {'name':'Africa', 'bonus':3, 'countries':['NAF','EGY']}
  ]

links = [('PER','VEN'),
         ('PER','BRA'),
         ('BRA','VEN'),
         ('VEN','MEX'),
         ('MEX','GRE'),
         ('GRE','ICE'),
         ('ICE','SEU'),
         ('SEU','NAF'),
         ('SEU','EGY'),         
         ('NAF','BRA'),
         ('NAF','EGY'),         
         ]

#%% Imports

import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import matplotlib.pyplot as plt
from game import MapLoader

#%% Create & save

path = '../support/maps/test_map.json'

# Create the classic world map from Risk
G = nx.Graph()
G.add_edges_from(links)
graph_data = json_graph.node_link_data(G.to_directed())

whole_file = {"map_graph": graph_data, "countries": countries, "continents":continents}

if os.path.exists(path):
  os.remove(path)
  
# Save the file in the json format designed for the MapLoader

with open(path,'w') as f:
  json.dump(whole_file, f)



#%% Test the MapLoader class to load the map

# MapLoader
mapLoader = MapLoader(path)
mapLoader.load_from_json()


# Create a DataFrame for visualization
df = pd.DataFrame(data = mapLoader.countries).T

df['cont_name'] = df['continent'].apply(lambda i: mapLoader.continents[i]['name'])

df['angle'] = df['continent'].apply(lambda i: np.pi*2*i/6)
df.set_index('continent', inplace=True, drop=True)

df['1'] = 1

gb_count = df.loc[:,['id']].groupby(level=0).agg({'id':'count'}).rename(columns={'id':'count'})  
df = df.join(gb_count)

gb_sub = df.loc[:,['1']].groupby(level=0).agg({ '1':'cumsum'})
df['sub_ind'] = gb_sub['1']

df['main_vector'] = df['angle'].apply(lambda x: np.array([np.cos(x), np.sin(x)]))

df['angle'] = np.pi*2* df['sub_ind']/ df['count']
df['sub_vector'] = df['angle'].apply(lambda x: np.array([np.cos(x), np.sin(x)]))

df['pos'] = df['main_vector']*1.0 + df['sub_vector']*0.3

pos = {n:df.loc[df['id']==mapLoader.countries[n]['id'],'pos'].item() for n in mapLoader.map_graph.nodes}

labels = {n:mapLoader.countries[n]['id'] for n in mapLoader.map_graph.nodes}


# Draw the map
if False:
  fig, ax = plt.subplots(1,1, figsize=(20,20))
  nx.draw_networkx(mapLoader.map_graph, with_labels=True, pos = pos,
                   labels=labels,
                   node_color='#def9ff',
                   node_size =600,
                   font_size =12,
                   font_color='black',
                   alpha=1.0, ax = ax)