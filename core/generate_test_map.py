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
  {'name':'South America', 'bonus':2, 'countries':['VEN','BRA','PER'], 'color':[252, 59, 45]},
  {'name':'North America', 'bonus':5, 'countries':['MEX', 'GRE'], 'color':[255, 176, 31]},
  {'name':'Europe', 'bonus':5, 'countries':['ICE', 'SEU'],  'color':[23, 236, 255]},
  {'name':'Africa', 'bonus':3, 'countries':['NAF','EGY'], 'color':[138, 70, 18]}
  
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
import networkx as nx
from networkx.readwrite import json_graph
import json
import os


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

