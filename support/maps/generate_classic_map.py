# Create the elements of the classic world map

countries = [
  # South America
  {'id': 'ARG', 'name':'Argentina', 'xy':'250,600', 'card_kind':1},
  {'id': 'VEN', 'name':'Venezuela', 'xy':'232,405', 'card_kind':2},
  {'id': 'BRA', 'name':'Brazil', 'xy':'330,470', 'card_kind':1},
  {'id': 'PER', 'name':'Peru', 'xy':'201,490', 'card_kind':2},

  #North America
  {'id': 'MEX', 'name':'Mexico', 'xy':'172,335', 'card_kind':1},
  {'id': 'EUS', 'name':'Eastern United Stated', 'xy':'130,260', 'card_kind':3},
  {'id': 'WUS', 'name':'Western United Stated', 'xy':'228,250', 'card_kind':3},
  {'id': 'QBC', 'name':'Quebec', 'xy':'272,172', 'card_kind':3},
  {'id': 'ONT', 'name':'Ontario', 'xy':'185,169', 'card_kind':3},
  {'id': 'WC', 'name':'Western Canada', 'xy':'86,168', 'card_kind':3},
  {'id': 'ALA', 'name':'Alaska', 'xy':'17,93', 'card_kind':3},
  {'id': 'NWT', 'name':'North West Territory', 'xy':'148,78', 'card_kind':2},
  {'id': 'GRE', 'name':'Greenland - Nunavut', 'xy':'330,60', 'card_kind':2},

  # Europe
  {'id': 'ICE', 'name':'Iceland', 'xy':'460,76', 'card_kind':2},
  {'id': 'GB', 'name':'Great Britain', 'xy':'441,171', 'card_kind':1},
  {'id': 'SCA', 'name':'Scandinavia', 'xy':'570,105', 'card_kind':1},
  {'id': 'UKR', 'name':'Ukraine', 'xy':'662,155', 'card_kind':1},
  {'id': 'NEU', 'name':'Northern Europe', 'xy':'551,190', 'card_kind':2},
  {'id': 'WEU', 'name':'Western Europe', 'xy':'479,275', 'card_kind':1},
  {'id': 'SEU', 'name':'Southern Europe', 'xy':'574,270', 'card_kind':1},

  # Africa
  {'id': 'NAF', 'name':'North Africa', 'xy':'461,407', 'card_kind':1},
  {'id': 'EGY', 'name':'Egypt', 'xy':'570,380', 'card_kind':2},
  {'id': 'CON', 'name':'Congo', 'xy':'532,502', 'card_kind':3},
  {'id': 'EAF', 'name':'East Africa', 'xy':'621,466', 'card_kind':1},
  {'id': 'SAF', 'name':'South Africa', 'xy':'573,603', 'card_kind':3},
  {'id': 'MAD', 'name':'Madagascar', 'xy':'703,553', 'card_kind':2},
  

  # Asia
  {'id': 'MDE', 'name':'Middle East', 'xy':'666,306', 'card_kind':3},
  {'id': 'AFG', 'name':'Afghanistan', 'xy':'763,242', 'card_kind':1},
  {'id': 'URA', 'name':'Ural', 'xy':'768,133', 'card_kind':1},
  {'id': 'SIB', 'name':'Siberia', 'xy':'859,101', 'card_kind':1},
  {'id': 'YAK', 'name':'Yakutsk', 'xy':'958,45', 'card_kind':3},
  {'id': 'IRK', 'name':'Irkutsk', 'xy':'956,122', 'card_kind':3},
  {'id': 'MON', 'name':'Mongolia', 'xy':'957,204', 'card_kind':2},
  {'id': 'CHI', 'name':'China', 'xy':'881,272', 'card_kind':3},
  {'id': 'IND', 'name':'India', 'xy':'772,356', 'card_kind':2},
  {'id': 'SIA', 'name':'Siam', 'xy':'883,361', 'card_kind':2},
  {'id': 'JAP', 'name':'Japan', 'xy':'1050,220', 'card_kind':2},
  {'id': 'KAM', 'name':'Kamchatka', 'xy':'1040,111', 'card_kind':3},

  # Oceania
  {'id': 'WAU', 'name':'Western Australia', 'xy':'938,545', 'card_kind':3},
  {'id': 'EAU', 'name':'Eastern Australia', 'xy':'1017,558', 'card_kind':2},
  {'id': 'INDO', 'name':'Indonesia', 'xy':'931,450', 'card_kind':1},
  {'id': 'NGU', 'name':'New Guinea', 'xy':'1042,460', 'card_kind':2}
  ]

continents = [
  {'name':'South America', 'bonus':2, 'countries':['ARG','VEN','BRA','PER'], 'color':[252, 59, 45]},
  {'name':'North America', 'bonus':5, 'countries':['MEX','EUS','WUS', 'ONT','QBC', 'WC', 'ALA', 'NWT', 'GRE'], 'color':[255, 176, 31]},
  {'name':'Europe', 'bonus':5, 'countries':['ICE','GB','SCA','UKR', 'NEU', 'WEU', 'SEU'], 'color':[23, 236, 255]},
  {'name':'Africa', 'bonus':3, 'countries':['NAF','EGY','CON','EAF', 'SAF', 'MAD'], 'color':[138, 70, 18]},
  {'name':'Asia', 'bonus':7, 'countries':['MDE','AFG', 'URA','SIB', 'YAK', 'IRK', 'MON', 'CHI', 'IND', 'SIA', 'JAP', 'KAM'], 'color':[0, 255, 42]},
  {'name':'Oceania', 'bonus':3, 'countries':['WAU','EAU','INDO','NGU'], 'color':[255, 59, 219]}
  ]

links = [('ARG','PER'),
         ('ARG','BRA'),
         ('PER','VEN'),
         ('PER','BRA'),
         ('BRA','VEN'),
         ('VEN','MEX'),
         ('MEX','WUS'),
         ('MEX','EUS'),
         ('EUS','WUS'),
         ('EUS','WC'),
         ('EUS','ONT'),
         ('WUS','ONT'),
         ('WUS','QBC'),
         ('WC','ALA'),
         ('WC','NWT'),
         ('WC','ONT'),
         ('ONT','QBC'),
         ('ONT','NWT'),
         ('ONT','GRE'),
         ('NWT','ALA'),
         ('NWT','GRE'),
         ('QBC','GRE'),
         ('GRE','ICE'),
         ('ICE','SCA'),
         ('ICE','GB'),
         ('GB','SCA'),
         ('GB','WEU'),
         ('GB','NEU'),
         ('SCA','NEU'),
         ('SCA','UKR'),
         ('UKR','URA'),
         ('UKR','AFG'),
         ('UKR','MDE'),
         ('UKR','SEU'),
         ('UKR','NEU'),
         ('NEU','SEU'),
         ('NEU','WEU'),
         ('WEU','SEU'),
         ('WEU','NAF'),           
         ('SEU','NAF'),
         ('SEU','EGY'),
         ('SEU','MDE'),
         ('NAF','BRA'),
         ('NAF','EGY'),
         ('NAF','EAF'),
         ('NAF','CON'),
         ('CON','EAF'),
         ('CON','SAF'),
         ('SAF','EAF'),
         ('SAF','MAD'),
         ('MAD','EAF'),
         ('EAF','EGY'),
         ('EAF','MDE'),
         ('EGY','MDE'),
         ('MDE','IND'),
         ('MDE','AFG'),
         ('AFG','IND'),
         ('AFG','CHI'),
         ('AFG','URA'),
         ('URA','SIB'),
         ('URA','CHI'),
         ('SIB','YAK'),
         ('SIB','IRK'),
         ('SIB','MON'),
         ('SIB','CHI'),
         ('YAK','KAM'),
         ('YAK','IRK'),
         ('IRK','KAM'),
         ('IRK','MON'),
         ('MON','KAM'),
         ('MON','CHI'),
         ('MON','JAP'),
         ('CHI','IND'),
         ('CHI','SIA'),
         ('SIA','IND'),
         ('SIA','INDO'),
         ('INDO','NGU'),
         ('INDO','WAU'),
         ('WAU','EAU'),
         ('WAU','NGU'),
         ('EAU','NGU'),
         ('JAP','KAM'),
         ('KAM','ALA'),                      
         ]

#%% Imports

import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import matplotlib.pyplot as plt
from pyRisk import MapLoader

#%% Create & save

path = '../support/maps/classic_world_map.json'

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