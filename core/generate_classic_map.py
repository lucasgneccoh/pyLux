# Create the elements of the classic world map

countries = [
  # South America
  {'id': 'ARG', 'name':'Argentina'},
  {'id': 'VEN', 'name':'Venezuela'},
  {'id': 'BRA', 'name':'Brazil'},
  {'id': 'PER', 'name':'Peru'},

  #North America
  {'id': 'MEX', 'name':'Mexico'},
  {'id': 'EUS', 'name':'Eastern United Stated'},
  {'id': 'WUS', 'name':'Western United Stated'},
  {'id': 'QBC', 'name':'Quebec'},
  {'id': 'ONT', 'name':'Ontario'},
  {'id': 'WC', 'name':'Western Canada'},
  {'id': 'ALA', 'name':'Alaska'},
  {'id': 'NWT', 'name':'North West Territory'},
  {'id': 'GRE', 'name':'Greenland - Nunavut'},

  # Europe
  {'id': 'ICE', 'name':'Iceland'},
  {'id': 'GB', 'name':'Great Britain'},
  {'id': 'SCA', 'name':'Scandinavia'},
  {'id': 'UKR', 'name':'Ukraine'},
  {'id': 'NEU', 'name':'Northern Europe'},
  {'id': 'WEU', 'name':'Western Europe'},
  {'id': 'SEU', 'name':'Southern Europe'},

  # Africa
  {'id': 'NAF', 'name':'North Africa'},
  {'id': 'EGY', 'name':'Egypt'},
  {'id': 'CON', 'name':'Congo'},
  {'id': 'EAF', 'name':'East Africa'},
  {'id': 'SAF', 'name':'South Africa'},
  {'id': 'MAD', 'name':'Madagascar'},
  

  # Asia
  {'id': 'MDE', 'name':'Middle East'},
  {'id': 'AFG', 'name':'Afghanistan'},
  {'id': 'URA', 'name':'Ural'},
  {'id': 'SIB', 'name':'Siberia'},
  {'id': 'YAK', 'name':'Yakutsk'},
  {'id': 'IRK', 'name':'Irkutsk'},
  {'id': 'MON', 'name':'Mongolia'},
  {'id': 'CHI', 'name':'China'},
  {'id': 'IND', 'name':'India'},
  {'id': 'SIA', 'name':'Siam'},
  {'id': 'JAP', 'name':'Japan'},
  {'id': 'KAM', 'name':'Kamchatka'},

  # Oceania
  {'id': 'WAU', 'name':'Western Australia'},
  {'id': 'EAU', 'name':'Eastern Australia'},
  {'id': 'INDO', 'name':'Indonesia'},
  {'id': 'NGU', 'name':'New Guinea'}
  ]

continents = [
  {'name':'South America', 'bonus':2, 'countries':['ARG','VEN','BRA','PER']},
  {'name':'North America', 'bonus':5, 'countries':['MEX','EUS','WUS', 'ONT','QBC', 'WC', 'ALA', 'NWT', 'GRE']},
  {'name':'Europe', 'bonus':5, 'countries':['ICE','GB','SCA','UKR', 'NEU', 'WEU', 'SEU']},
  {'name':'Africa', 'bonus':3, 'countries':['NAF','EGY','CON','EAF', 'SAF', 'MAD']},
  {'name':'Asia', 'bonus':7, 'countries':['MDE','AFG', 'URA','SIB', 'YAK', 'IRK', 'MON', 'CHI', 'IND', 'SIA', 'JAP', 'KAM']},
  {'name':'Oceania', 'bonus':3, 'countries':['WAU','EAU','INDO','NGU']}
  ]

links = [('ARG','PER'),
         ('ARG','BRA'),
         ('PER','VEN'),
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


import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import matplotlib.pyplot as plt
from game import MapLoader


# Create the classic world map from Risk
G = nx.Graph()
G.add_edges_from(links)
graph_data = json_graph.node_link_data(G.to_directed())

whole_file = {"map_graph": graph_data, "countries": countries, "continents":continents}

if os.path.exists('classic_world_map.json'):
  os.remove('classic_world_map.json')
  
# Save the file in the json format designed for the MapLoader

with open('classic_world_map.json','w') as f:
  json.dump(whole_file, f)



#%% Test the MapLoader class to load the map

# MapLoader
mapLoader = MapLoader('classic_world_map.json')
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
if True:
  fig, ax = plt.subplots(1,1, figsize=(20,20))
  nx.draw_networkx(mapLoader.map_graph, with_labels=True, pos = pos,
                   labels=labels,
                   node_color='#def9ff',
                   node_size =600,
                   font_size =12,
                   font_color='black',
                   alpha=1.0, ax = ax)