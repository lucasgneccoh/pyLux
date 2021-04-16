


# To define here

# Board
# Card
# Country
# Move (Maybe? depends on the phase)
# World: Lux uses world to keep the game state.
# Board contains the methods that Agents can use to interact with the game


import networkx as nx
from networkx.readwrite import json_graph
import json
import os

##%% MapLoader
'''
Class used to read a file (format to define) and yield a \
nx.DiGraph representing a map.
'''

class MapLoader(object):
  '''
  Creates a graph representing the world. Nodes are countries.
  An auxiliary dict of continents is also read to get the \
  different bonuses
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
    
    
    
  '''
  def __init__(self, path):
    self.path = path

    self.continents = None
    self.map_graph = None
    self.countries = None
    
    
  def load_from_json(self):
    try:
      with open(self.path,'r') as f:
        data = json.load(f)
      self.map_graph = nx.DiGraph(json_graph.node_link_graph(data['map_graph']))
      continents = data['continents']
      countries = data['countries']
      # Create the actual countries and continents
      aux_countries = {c['id']: Country(**c) for c in countries}
      self.continents = {i: Continent(**c) for i,c in enumerate(continents)}
      for k, cont in self.continents.items():
        for c in cont['countries']:
          aux_countries[c]['continent'] = k
      
      self.countries = {i: aux_countries[k] for i,k in enumerate(aux_countries)}
      relabel = {c['id']:i for i, c in self.countries.items()}
      nx.relabel_nodes(self.map_graph, relabel, copy=False)
      
    except Exception as e:
      print("MapLoader: Problems while loading map")
      print(e)


##%% World

'''
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

class World(object):
  '''
  Represents the world as a graph
  map_graph: nx.DiGraph
  '''
  def __init__(self, mapLoader:MapLoader):
    mapLoader.load_from_json()    
    self.map_graph = mapLoader.map_graph # nx.Digraph
    self.continents = mapLoader.continents # dict by continent
    self.coutries = mapLoader.countries
    


##%% Continent
class Continent(dict):
  def __init__(self, **kwargs):
    super().__init__(kwargs)
    



##%% Country

'''
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
'''
class Country(dict):
  def __init__(self, **kwargs):
    super().__init__(kwargs)    
##    self.armies = -1
##    self.movable_armies = -1
##    self.world = None # nx.DiGraph modelling the world map
    
##  '''  Returns the country-code of this Country.  '''
##  def getCode(self) -> int:
##    return self.code
##  
##  
##  '''  Returns the current owner-code of this Country.  '''
##  def getOwner(self) -> int:  
##    return self.owner.code if not self.owner is None else -1
##  
##
##  '''  Returns the continent-code of this Country.  '''
##  def getContinent(self) -> int: 
##    return self.continent.code;
##  
##
##  '''  Returns the number of armies in this Country.  '''
##  def getArmies(self) -> int:
##    return self.armies;
##  
##  
##  '''  Returns the number of armies in this Country that may
##  be fortified somewhere else. This is only garanteed to be
##  accurate during the fortifyPhase of each Agent.  '''  
##  def getMoveableArmies(self) -> int:
##    return self.movable_armies;
##  
##  
##  '''  Returns an array containing all of the Country's that
##  are touching this Country. They might not be reachable.
##  Use canGoTo to test'''
##  
##  def getAdjoiningList(self):
##    return self.neighbors;
##  
##  
##  ''' An adjacency test: Note that is it possible for maps to
##  contain one-way connections. Thus a.canGoto(b) will NOT always
##  return the same result as b.canGoto(a). '''
##  def canGoto(self, other) -> bool:
##    # For now this seems like a long workaround by calling world,
##    # but I'll keep it so that the code remains similar to the
##    # original Java Lux
##    # Remember that world is a nx.DiGraph
##    # self.world.
##    return False
##    
##  
##  '''  Depreciated: use canGoto() instead.
##  This method will not behave correctly when used on a map that
##  has one-way connections. Use the canGoto() methods instead. '''
##  def isNextTo(self, other) -> bool:
##    return False
##  
##
##  ''' Returns a reference to the weakest neighbor that is owned
##  by another player, or null if there are no enemy neighbors.'''
##  def getWeakestEnemyNeighbor(self):
##    return None
##  
##  ''' Operates like getWeakestEnemyNeighbor but limits its search
##  to the given continent. '''
##  def getWeakestEnemyNeighborInContinent(self, cont:int):
##    return None
##  
##  
##  ''' Returns the number of adjacent countries. '''
##  def getNumberNeighbors(self) -> int:
##    return -1
##  
##
##  ''' Returns the number of neighbor countries owned by players
##  that don't own this Country. '''
##  def getNumberEnemyNeighbors(self) -> int:
##    return -1
##
##  ''' Returns the number of adjacent countries owned by *player*.'''
##  def getNumberPlayerNeighbors(self, player:int) -> int:
##    return -1
##  
##  ''' Returns the number of adjacent countries not owned by  *player*.'''
##  def getNumberNotPlayerNeighbors(self, player:int) -> int:
##    return -1
##  
##  ''' Return the name of the country. This will only return the
##  correct value for maps that explicitly set country names. '''  
##  def getName(self) -> str:
##    return self.name
##  
##  ''' Returns a String representation of the country.  '''
##  def __repr__(self) -> str:
##    return 'text'
##
##  ''' Return an int array containing the country codes of all the
##  neigbors of this country that are owned by the same player.
##  Useful to determine where this country can fortify to. '''
##  def getFriendlyAdjoiningCodeList(self):
##    return [-1]
##
##  ''' Returns a reference to the neighbor that is owned by *player*
##  with the most number of armies on it. If none exist then return None. '''
##  def getStrongestNeighborOwnedBy(self, player:int):
##    return None
  
  ''' Returns the number of adjacent countries not owned by an agent
  named *agentName*.'''
  #def getNumberPlayerNotNamedNeighbors(self, agentName:str, board:Board) -> int:
  #  return -1

  ''' Return an int array containing the country codes of all the
  countries that this country can attack. '''
  #def[] getHostileAdjoiningCodeList(self)
  #return null;
  
  
  ''' Return an int array containing the country codes of all adjoining
  countries. '''
  #def[] getAdjoiningCodeList() {
  #return null;
  #}

  ''' Create a new country. The passkey object must be suplied to make
  any changes to the object. So you can only change Country objects that
  you creats, and not the ones the Board sends you. '''
  #def(int newCountryCode, int newContCode, Object passkey)
  #{}

  ''' Sets the continent code of the Country, as long as the passkey
  object is the same as supplied in the constructor. '''
  #def setContinentCode(int newContinentCode, Object passkey)
  #{}

  ''' Sets the owner code of the Country, as long as the passkey object
  is the same as supplied in the constructor. '''
  #def setOwner( int newOwner, Object passkey)
  #{}

  ''' Sets the number of armies on the Country, as long as the
  passkey object is the same as supplied in the constructor. '''
  #def setArmies( int newArmies, Object passkey)
  #{}
  
  ''' Adds one to the army count of the the Country, as long as
  the passkey object is the same as supplied in the constructor. '''
  #def addArmy(Object passkey)
  #{}


  ''' Add a connection from this Country object to the
  destinationCountry object. To be traversable both ways, the
  connection should be added in reverse as well. '''
  #def addToAdjoiningList( Country destinationCountry, Object #passkey )
  #{}

  ''' Add a 2-way connection between this Country object and the
  otherCountry object. '''  
  #def addToAdjoiningListBoth( Country otherCountry, Object passkey )
  #{}

  ''' Set the name of the Country. '''    
  #def setName(String name, Object passkey)
  #{}

  #def clearAdjoiningList(Object passkey)
  #{}


if __name__ == '__main__':
  # Testing

  pass
   
  
