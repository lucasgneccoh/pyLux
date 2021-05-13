# -*- coding: utf-8 -*-

import json
import networkx as nx
from networkx.readwrite import json_graphimport networkx as nx
import copy

#%% World, countries and continents
class World(object):
  '''!Creates a graph representing the world. Nodes are countries.
  An auxiliary dict of continents and countries is also read to get all the other information
  For now JSON is the only file format supported
  countries:

  JSON file structure:
    {
      "map_graph":{
        "directed": True in general. If set to False, internally
          it will be a digraph with edges in both directions
        "multigraph": False
        "graph": Dict with graph attributes. For the moment they are not used
        "nodes": List of node Dict with node ids.
                This id is the one you should use everywhere
                in this file to refer to this country.
                Ex: [{'id':'USA'}, {'id':'COL'}],
                or numbers like [{'id':1}, {'id':2}].
                It is completely up to the map creator.

                This id is also used to determine the edges of the map and continents
                
        "links": List of Dict
                Each Dict contains the information of the edge:
                'source': id of the source node
                'target': id of the target node
                Ex: [{'source':'USA', 'target':'COL'}]
      },        
      "countries": {
         List of countries. Each country is a dict containing the following fields
           'id': Same id used in the map graph to identify this country/node
           'name': Name of the country
           'code': (optional) Integer to represent the country.
        Ex: [{'id':'USA', 'name':'United States', 'code':3}, {'id':'COL', 'name':'Colombia', 'code':5}]

      },
      "continents": {
        List of continents. Each continent is a dict containing the following fields           
           'name': Name of the continent
           'code': (optional) Integer to represent the continent.
           'countries': A list of countries that belong to this continent.
            The elements of the list must be the ids of the countries used before
           'bonus': The bonus this continent gives to the player that owns all
              the countries composing it
        Ex: [{'name':'America', 'code':0, 'countries':['USA', 'COL'], 'bonus':5}]
      }
  }

  NOTE: The easiest way to create a map graph is to create a nx.Digraph
        and then save it to JSON. It will create the right structure.
        For example:
            G = nx.DiGraph(map_name = 'example')
            G.add_edges_from([('C1', 'C2'),
                              ('C2', 'C3'),
                              ('C2', 'C1')])
            
            data = json_graph.node_link_data(G)
            with open('json_test.json','w') as f:
              json.dump(data, f)
    
    
    

  The World object is a nx.DiGraph modelling the adjacency of \
  countries. Each node will be identified using the country id,\
  and will contain as attribute the Country object, so that it \
  can access all the County information.
  World will answer questions about adjancecy and "canGoTo". \
  Each Country should have a way to call the world to give this \
  information when needed.
  World may also contain metadata, for example dictionaries \
  with continents, bonuses, etc. to get information quickly.
  '''
  def __init__(self, path, load = True):
    '''!
    
    Parameters
    -----------------
    mapLoader: MapLoader
      Object to get the continent and countries information
    '''
    self.path = path
    if load:
      self.load_from_json()
      
  @staticmethod
  def fromDicts(continents:dict, countries:dict, inLinks:dict):    
    world = World(path = None, load = False)
    # Graph
    graph = nx.DiGraph()
    graph.add_nodes_from([i for i in countries])
    for k, v in inLinks.items():
      graph.add_edges_from([(x,k) for x in v])
    world.map_graph = graph
    # Continents
    world.continents = {k: Continent(**c.__dict__) for k, c in continents.items()}
    # Countries
    world.countries = {k: Country(**c.__dict__) for k, c in countries.items()}
    return world
    
  def toDicts(self):
    countries = {k: v for k, v in self.countries.items()}
    continents = {k: v for k, v in self.continents.items()}
    inLinks = {n:self.predecessors(n) for n in self.countries}
    return continents, countries, inLinks
    
    
    
    
  def load_from_json(self):
    '''!
    Read the JSON file and define the map_graph, countries and continents attributes

    Returns
    -------
    None.

    '''
    try:
      with open(self.path,'r') as f:
        data = json.load(f)
      self.map_graph = nx.Graph(json_graph.node_link_graph(data['map_graph'])).to_directed()
      continents = data['continents']
      countries = data['countries']
      # Create the actual countries and continents
      aux_countries = {c['id']: Country(**c, code=i) for i,c in enumerate(countries)}
      
      self.continents = {i: Continent(**c) for i,c in enumerate(continents)}
      for k, cont in self.continents.items():
        cont['countries_id'] = [ID for ID in cont['countries']]
        cont['countries'] = []
        for c in cont['countries_id']:
          cont['countries'].append(aux_countries[c].code)
          aux_countries[c]['continent'] = k
          
      self.countries = {v.code: v for _,v in aux_countries.items()}
      
      # Update attributes from keyword values
      for _, c in self.countries.items(): c.dictToAttr()
      for _, c in self.continents.items(): c.dictToAttr()
      
      relabel = {c.id: c.code for _, c in self.countries.items()}
      nx.relabel_nodes(self.map_graph, relabel, copy=False)
      
      # new_data = {c.code: c for c in self.countries.values()}
      # nx.set_node_attributes(self.map_graph, new_data)
      
    except Exception as e:
      print("World: Problems while loading map")
      raise e
      
  
  # Continent functions
  def isEmptyContinent(self, cont:int) -> bool:
    for i, c in self.continents.items():
      if c.owner != -1: return False
    return True
    
  def isOpenContinent(self, cont:int) -> bool:
    for i, c in self.continents.items():
      if c.owner != -1: return True
    return False
    
  # Country functions
  def successors(self, country:int):
    '''! Returns a list of countries accessible from the current country
    '''
    return [self.countries[i] for i in self.map_graph.successors(country)]
  
  def predecessors(self, country:int):
    '''! Returns a list of countries that can access the current country
    '''
    return [self.countries[i] for i in self.map_graph.predecessors(country)]

  def getAdjoiningList(self, country:int, kind=1):
    '''!  Returns an array containing all of the Country's that
    are touching this Country. They might not be reachable.
    Use canGoTo to test
    Use kind to get incoming edges, outgoing, or both
    kind = -1: Countries that can attack this country (predecessors)
    kind = 1: Countries that can be attacked by this country (successors)
    kind = 0: Both
    '''
    if kind == 1:
      return self.successors(country)
    elif kind == -1:
      return self.predecessors(country)
    else:
      return list(set(itertools.chain(self.successors(country), self.predecessors(country))))
      
  def getWeakestEnemyNeighbor(self, country:int, kind=1):
    '''! Returns a reference to the weakest neighbor that is owned by another player, or None if there are no enemy neighbors.'''
    m, weakest = sys.maxsize, None
    for c in self.getAdjoiningList(country, kind=kind):
      if c.owner != self.code and c.armies<m:
        m = c.armies
        weakest = c
    return weakest
    
  def getWeakestEnemyNeighborInContinent(self, country:int, cont:int, kind=1):
    '''! Operates like getWeakestEnemyNeighbor but limits its search to the given continent. '''
    m, weakest = sys.maxsize, None
    owner = self.countries[country].owner
    for c in self.getAdjoiningList(country, kind=kind):
      if c.continent == cont and c.owner != owner and c.armies<m:
        m = c.armies
        weakest = c
    return weakest
  
    
  def getNumberNeighbors(self, country:int, kind=0) -> int:
    '''! Returns the number of adjacent countries. '''
    return len(self.getAdjoiningList(country, kind))

  def getNumberEnemyNeighbors(self, country:int, kind=0) -> int:
    '''! Returns the number of neighbor countries owned by players that don't own this Country. '''
    s = 0
    owner = self.countries[country].owner
    for c in self.getAdjoiningList(country, kind=kind):
      if c.owner != owner: s += 1
    return s
    
  def getNumberPlayerNeighbors(self, country:int, player:int, kind=0) -> int:
    '''! Returns the number of adjacent countries owned by *player*.'''
    s = 0
    for c in self.getAdjoiningList(country, kind=kind):
      if c.owner == player: s += 1
    return s
  
  def getNumberNotPlayerNeighbors(self,country:int, player:int, kind=0) -> int:
    '''! Returns the number of adjacent countries not owned by  *player*.'''
    s = 0
    for c in self.getAdjoiningList(country,kind=kind):
      if c.owner != player: s += 1
    return s
    
  def getCountriesToAttack(self, source:int):
    '''! Return a list of all the countries that this source country can attack. '''    
    owner = self.countries[source].owner
    return [c for c in self.getAdjoiningList(source, kind=1) if c.owner != owner]
  
  def getCountriesToFortify(self, source:int):
    '''! Return a list of all the countries that this country can fortify. '''
    owner = self.countries[source].owner
    return [c for c in self.getAdjoiningList(source, kind=1) if c.owner == owner]    
  
  
  def __deepcopy__(self, memo):
    new_world = World(self.path, load=False)
    new_world.map_graph = copy.deepcopy(self.map_graph)
    new_world.countries = copy.deepcopy(self.countries)
    new_world.continents = copy.deepcopy(self.continents)
    
    return new_world
      

class Attr_dict(dict):
  '''! A dict with values also accessible using the object attribute notation
  '''
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)    
    for k,v in kwargs.items(): setattr(self, k, v)
  
  def dictToAttr(self):
    for k,v in self.items():
      setattr(self, k, v)


class Continent(Attr_dict):
  '''! Represents a continent
    Must have at least the following fields
      - code
      - bonus
  '''
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)
    if not hasattr(self, 'owner'): self.owner = -1
  
  def __repr__(self):
    return f'{self.name}. bonus = {self.bonus}, owner = {self.owner}'
    

class Country(Attr_dict):
  '''!
  A Country instance represents a single territory in the game.
  Each Country stores a country-code, continent-code, owner-code,
  number of armies, and number of fortifyable armies.  As well,
  each Country stores an array containing references to all adjoining 
  Country's.
  When first initialized by the game world, each Country will be given
  a permanent country-code, continent-code and adjoining-list. 
  The owner-code and number of armies are set to -1.
  The country-code is a unique number used to identify countries.
  The array returned by the Board.getCountries() will always be ordered \
  by country-code.
    Must have at least the following fields
      - code
      - name
      - continent
  '''
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)
    
    if not hasattr(self, 'armies'): self.armies = 0
    if not hasattr(self, 'moveableArmies'): self.moveableArmies = 0
    if not hasattr(self, 'owner'): self.owner = -1
    if not hasattr(self, 'continent'): self.continent = -1
    
  def __hash__(self):
    '''! Countries will be completely caracterized by their code
    '''
    return self.code
  
  def __eq__(self, o):
    '''! Countries will be completely caracterized by their code
    '''
    return self.code == o.code
  
  def __repr__(self) -> str:
    '''! Returns a String representation of the Country.  '''
    return f'{self.id} - {self.name}. owner = {self.owner}, armies = {self.armies}'

  def encode(self) -> str:
    return f'{self.id}_{self.owner}_{self.armies}'
  
  def addArmies(self, armies):
    '''! Adds armies to the army count of the the Country'''
    self.armies = self.armies + armies

