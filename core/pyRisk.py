


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
# import os
import sys
import itertools
import numpy as np
import copy
import random
import time


import agent
# For the GUI
import pygame


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
    '''
    Parameters
    ----------
    path : str
      Path to the JSOn file containing the map information, continent and countries.

    Returns
    -------
    None.

    '''
    self.path = path

    self.continents = None
    self.map_graph = None
    self.countries = None
    
    
  def load_from_json(self):
    '''
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
          cont['countries'].append(aux_countries[c])
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
    '''
    Parameters
    -----------------
    mapLoader: MapLoader
      Object to get the continent and countries information
    '''
    mapLoader.load_from_json()    
    self.map_graph = mapLoader.map_graph # nx.Digraph
    self.continents = mapLoader.continents # dict by continent
    self.countries = mapLoader.countries
    for _, c in self.countries.items():
      c.setWorld(self)
    
  
##%% Dict-like with attributes
class Attr_dict(dict):
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)    
    for k,v in kwargs.items(): setattr(self, k, v)
  
  def dictToAttr(self):
    for k,v in self.items():
      setattr(self, k, v)


##%% Continent
class Continent(Attr_dict):
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)
    self.owner = -1
    
  def isEmpty(self):
    for c in self.countries:
      if c.owner != -1: return False
    return True
  
  def isOpen(self):
    for c in self.countries:
      if c.owner == -1: return True
    return False
  
  def updateOwner(self):
    p = self.countries[0].owner
    for c in self.countries:
      if p != c.owner:
        self.owner = -1
        return
        
    self.owner = p
      
      
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
class Country(Attr_dict):
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)
    self.armies = 0
    self.movable_armies = 0
    self.owner = -1
    self.continent = -1
    self.world = None # World
    
  def __hash__(self):
    return self.code
  
  def __eq__(self, o):
    return self.code == o.code

  
  def setWorld(self, world):
    self.world = world
    inN = list(world.map_graph.successors(self.code))
    outN = list(world.map_graph.predecessors(self.code))
    self.inNeighbors = len(inN)
    self.outNeighbors = len(outN)
    self.totalNeighbors = len(set(inN+outN))
    
  
  def successors(self): 
    return [self.world.countries[i] for i in self.world.map_graph.successors(self.code)]
      
    
  
  def predecessors(self):
    return [self.world.countries[i] for i in self.world.map_graph.predecessors(self.code)]
  
  '''  Returns the country-code of this Country.  '''
  def getCode(self) -> int:
    return self.code
  
  
  '''  Returns the current owner-code of this Country.  '''
  def getOwner(self) -> int:  
    return self.owner.code if not self.owner is None else -1
  

  '''  Returns the continent-code of this Country.  '''
  def getContinent(self) -> int: 
    return self.continent.code;
  

  '''  Returns the number of armies in this Country.  '''
  def getArmies(self) -> int:
    return self.armies;
  
  
  '''  Returns the number of armies in this Country that may
  be fortified somewhere else. This is only garanteed to be
  accurate during the fortifyPhase of each Agent.  '''  
  def getMoveableArmies(self) -> int:
    return self.movable_armies;
  
  
  '''  Returns an array containing all of the Country's that
  are touching this Country. They might not be reachable.
  Use canGoTo to test
  Use kind to get incoming edges, outgoing, or both
  kind = -1: Countries that can attack this country
  kind = 1: Countries that can be attacked by this country
  kind = 0: All
  '''
  def getAdjoiningList(self, kind=1):
    if kind == 1:
      return self.successors()
    elif kind == -1:
      return self.predecessors()
    else:
      return itertools.chain(self.successors(), self.predecessors())
  
  
  ''' An adjacency test: Note that is it possible for maps to
  contain one-way connections. Thus a.canGoto(b) will NOT always
  return the same result as b.canGoto(a). '''
  def canGoto(self, other) -> bool:
    # For now this seems like a long workaround by calling world,
    # but I'll keep it so that the code remains similar to the
    # original Java Lux
    # Remember that world is a nx.DiGraph
    # self.world.
    return other.code in self.successors()
    
  
  '''  Depreciated: use canGoto() instead.
  This method will not behave correctly when used on a map that
  has one-way connections. Use the canGoto() methods instead. '''
  def isNextTo(self, other) -> bool:
    return other.code in itertools.chain(self.successors(), self.predecessors())
  

  ''' Returns a reference to the weakest neighbor that is owned
  by another player, or null if there are no enemy neighbors.'''
  def getWeakestEnemyNeighbor(self, kind=1):
    m, weakest = sys.maxsize, None
    for c in self.getAdjoiningList(kind=kind):
      if c.owner != self.code and c.armies<m:
        m = c.armies
        weakest = c
    return weakest
  
  ''' Operates like getWeakestEnemyNeighbor but limits its search
  to the given continent. '''
  def getWeakestEnemyNeighborInContinent(self, cont:int, kind=1):
    m, weakest = sys.maxsize, None
    for c in self.getAdjoiningList(kind=kind):
      if c.continent == cont and c.owner != self.code and c.armies<m:
        m = c.armies
        weakest = c
    return weakest
  
  
  ''' Returns the number of adjacent countries. '''
  def getNumberNeighbors(self) -> int:
    return self.totalNeighbors
  

  ''' Returns the number of neighbor countries owned by players
  that don't own this Country. '''
  def getNumberEnemyNeighbors(self, kind=0) -> int:
    s = 0
    for c in self.getAdjoiningList(kind=kind):
      if c.owner != self.owner: s += 1
    return s

  ''' Returns the number of adjacent countries owned by *player*.'''
  def getNumberPlayerNeighbors(self, player:int, kind=0) -> int:
    s = 0
    for c in self.getAdjoiningList(kind=kind):
      if c.owner == player: s += 1
    return s
  
  ''' Returns the number of adjacent countries not owned by  *player*.'''
  def getNumberNotPlayerNeighbors(self, player:int, kind=0) -> int:
    s = 0
    for c in self.getAdjoiningList(kind=kind):
      if c.owner != player: s += 1
    return s
  
  ''' Return the name of the country. This will only return the
  correct value for maps that explicitly set country names. '''  
  def getName(self) -> str:
    return self.name
  
  ''' Returns a String representation of the country.  '''
  def __repr__(self) -> str:
    return str(self.__dict__)

  ''' Return an int array containing the country codes of all the
  neigbors of this country that are owned by the same player.
  Useful to determine where this country can fortify to. '''
  def getFriendlyAdjoiningCodeList(self, kind=1):
    s = []
    for c in self.getAdjoiningList(kind=kind):
      if c.owner == self.code: s += [c.code]
    return s

  ''' Returns a reference to the neighbor that is owned by *player*
  with the most number of armies on it. If none exist then return None. '''
  def getStrongestNeighborOwnedBy(self, player:int, kind=0):
    m, strongest = -1, None
    for c in self.getAdjoiningList(kind=kind):
      if c.owner == player and c.armies>m:
        m = c.armies
        strongest = c
    return strongest
  
  ''' Returns the number of adjacent countries not owned by an agent
  named *agentName*.'''
  def getNumberPlayerNotNamedNeighbors(self, agentName:str, board) -> int:
    code = -1
    for i, p in board.players.items():
      if p.name() == agentName: 
        code = i
        break
    if code == -1: return None
    return self.getNumberNotPlayerNeighbors(code)

  ''' Return an int array containing the country codes of all adjoining
  countries. '''
  
  def getAdjoiningCodeList(self, kind=1):
    return [c.code for c in self.getAdjoiningList(kind=kind)]
  
  ''' Return an int array containing the country codes of all the
  countries that this country can attack. '''
  def getHostileAdjoiningCodeList(self):
    return [c.code for c in self.getAdjoiningList(kind=1) if c.owner != self.owner]
  
  
  ''' Adds one to the army count of the the Country, as long as
  the passkey object is the same as supplied in the constructor. '''
  def addArmies(self, armies):
    self.armies = self.armies + armies
  

class AbstractCardSequence(object):
  def __init__(self):
    self.cont = 0
  
  def nextCashArmies(self):
    self.cont += 1
    return self.cont


class GeometricCardSequence(AbstractCardSequence):
  def __init__(self, base = 6, incr = 0.03):
    super().__init__()
    self.base = base
    self.incr = incr
    self.factor = 1.0
  
  def nextCashArmies(self):
    self.cont += 1
    self.factor *= (1+self.incr)
    return int(self.base*self.factor)

class ArithmeticCardSequence(AbstractCardSequence):
  def __init__(self, base = 4, incr = 2):
    super().__init__()
    self.base = base
    self.incr = incr    
  
  def nextCashArmies(self):
    self.cont += 1
    return int(self.base + self.cont*self.incr)


class ListCardSequence(AbstractCardSequence):
  def __init__(self, sequence):
    super().__init__()
    self.sequence = sequence
    self.M = len(sequence)-1
  
  def nextCashArmies(self):
    res = self.sequence[min(self.cont, self.M)]
    self.cont += 1
    return res

class ListThenArithmeticCardSequence(AbstractCardSequence):
  def __init__(self, sequence, incr):
    super().__init__()
    self.sequence = sequence
    self.M = len(sequence)-1
    self.incr = incr    
    self.arithmetic = sequence[-1]
  
  def nextCashArmies(self):
    if self.cont <= self.M:
      res = self.sequence[self.cont]
    else:
      self.arithmetic += self.incr
      res = self.arithmetic
    self.cont += 1
    return res


#%% Card
class Card(object):
  # kind = 0 means wildcard 
  def __init__(self, code, kind):
    if kind == 0 and code >= 0:
      raise Exception(f"Wildcard (kind=0) can not have non negative code ({code})")
    self.code = code
    self.kind = kind
  
  def __eq__(self, o):
    if self.kind==0 and o.kind==0:
      return False
    if self.kind==0 or o.kind==0:
      return True
    return self.kind == o.kind
    
  def __ne__(self, o):
    if self.kind==0 and o.kind==0:
      return False
    if self.kind==0 or o.kind==0:
      return True
    return self.kind != o.kind
    
  def __hash__(self):
    return self.code
  
  def __repr__(self):
    return str(self.code)

class Deck(object):
  def __init__(self):
    self.deck = []
  
  def shuffle(self):
    random.shuffle(self.deck)
  
  def create_deck(self, countries, num_wildcards = 2):
    for c in countries:
      self.deck.append(Card(c.code, c.card_kind))
    for i in range(num_wildcards):
      self.deck.append(Card(-i-1, 0))    
    self.orig_deck = [card for card in self.deck]
    self.shuffle()
    
  
  def draw(self):
    card = self.deck.pop(0)
    if len(self.deck)==0:
      # Restart cards for now, to avoid problems
      # We can add a variable to stop drawing cards when done
      self.deck = [card for card in self.orig_deck]
      self.shuffle()
    return card
  
  @staticmethod
  def isSet(c1,c2,c3):
    if c1==c2 and c2==c3 and c1==c3: 
      return True
    if c1!=c2 and c2!=c3 and c1!=c3: 
      return True
    return False
  
  @staticmethod
  def containsSet(deck):
    if len(deck)<3: return False
    for c in itertools.combinations(deck, 3):
      if Deck.isSet(*c): return True
    return False
  
  @staticmethod
  def yieldCashableSet(deck):
    if len(deck)<3: return None
    for c in itertools.combinations(deck, 3):
      if Deck.isSet(*c): return c
    return None
  
  @staticmethod
  def yieldBestCashableSet(deck, player_code, countries):
    # Look for cards of territories owned by the player
    best, best_s = None, -1
    if len(deck)<3: return None    
    for c in itertools.combinations(deck, 3):
      s = 0
      if Deck.isSet(*c):
        for card in c:
          if card.code >= 0 and countries[card.code].owner == player_code:            
            s += 1
        if best_s < s:
          best_s = s
          best = c        
    return best
  
    
  
  

#%% Board
class Board(object):
  
  def __init__(self, world, players):
  
    # Game setup
    self.world = world
    self.countries = [c for _,c in world.countries.items()]
    N = len(players)
    
    # This should come from the MapLoader
    # Because it will change in each map
    armiesForPlayers = {2:45, 3: 35, 4: 30, 5: 25, 6:20}
    #armiesForPlayers = {2:10, 3: 10, 4: 30, 5: 25, 6:20} # For testing
    if N > 6: 
      initialArmies = 20
    elif N < 2:
      raise Exception("Minimum 2 players to play pyRisk")
    else:
      initialArmies = armiesForPlayers[N]
    
    # random.shuffle(players)
    self.players = {i: p for i, p in enumerate(players)}
    for i, p in self.players.items():
      p.setPrefs(i, self)
      
    self.startingPlayers = N    
    self.nextCashArmies = 0 
    self.tookOverCountry = False
    self.turnCount = 0
    self.countriesLeft = [c for c in self.countries]
    
    
    self.playerCycle = itertools.cycle(list(self.players.values()))
    self.activePlayer = next(self.playerCycle)
    self.firstPlayerCode = self.activePlayer.code
    self.lastPlayerCode = list(self.players.values())[-1].code
    self.cacheArmies = 0
    self.gamePhase = 'initialPick'
    
    
      
    
   
    
    # Preferences (Default here)
    self.initialPhase = True
    self.useCards = True
    self.transferCards = True
    self.immediateCash = False
    self.turnSeconds = 30
    self.continentIncrease = 0.05
    self.pickInitialCountries = False
    self.armiesPerTurnInitial = 4
    self.console_debug = False
    
    self.cardSequence = ListThenArithmeticCardSequence(sequence=[4,6,8,10,12,15], incr=5)
    self.deck = Deck()
    self.deck.create_deck(self.countries, num_wildcards=2)
    self.nextCashArmies = self.cardSequence.nextCashArmies()
    self.aux_cont = 0
    
    # Start players
    for _, p in self.players.items():
      p.is_alive = True
      p.cards = []
      p.income = 0
      p.initialArmies = int(initialArmies)
      p.num_countries = 0
    
  
  def setPreferences(self, prefs):
    available_prefs = ['initialPhase', 'useCards', 'transferCards', 'immediateCash', 'continentIncrease', 'pickInitialCountries', 'armiesPerTurnInitial', 'console_debug']
    # In the future, card sequences, num_wildcards
    for a in available_prefs:
      r = prefs.get(a)
      if r is None:
        print(f"Board preferences: value for '{a}' not given. Leaving default value {getattr(self, a)}")
      else:
        setattr(self, a, r)
  
  def randomInitialPick(self):
    
    player_codes = itertools.cycle(list(self.players.keys()))
    random.shuffle(self.countriesLeft)
    while self.countriesLeft:
      c = self.countriesLeft.pop()
      p = next(player_codes)
      c.owner = p
      c.addArmies(1)
      self.players[p].initialArmies -= 1
      self.players[p].num_countries += 1  
      if self.console_debug: print(f"Board:randomInitialPick: Picked {c.id} for {p.code} ({p.name()})")
    self.gamePhase = 'initialFortify'
    
  def randomInitialFotify(self):        
    over = False
    N = self.getNumberOfPlayers()
    
    # Get countries first to avoid getting them every time
    countries_players = {p.code: [c for k, c in self.world.countries.items() if c.owner == p.code] for i,p in self.players.items()}
    
    while not over:            
      # Check if there are armies left to put
      cont = 0
      while self.activePlayer.income==0 and self.activePlayer.initialArmies==0 and cont <= N:
        cont += 1
        self.endTurn()
        if cont >= N:
          # Prepare everything for the next round
          self.activePlayer = self.players[self.firstPlayerCode]
          over = True
      
      if over: break
      
      # Not over
      # Get income, put it in random countries
      p = self.activePlayer      
      armies = p.income
      if self.console_debug: print(f"Board:randomInitialFotify: For {p.code} ({p.name()})")
      for _ in range(armies):
        c = np.random.choice(countries_players[p.code])
        c.addArmies(1)
        if self.console_debug: print(f"{c.id} - {c.name}")
      
      p.income -= armies
      self.endTurn()
      
    self.gamePhase = 'startTurn'
  
  def initialPickOneHuman(self, country): 
    if isinstance(country, int):
      country = self.world.countries[country]
    p = self.activePlayer
    if not p.human: return # only made to wait human picks      
    if not country is None and country.owner == -1:
      country.owner = p.code
      country.addArmies(1)
      self.countriesLeft.remove(country)
      p.initialArmies -= 1
      p.num_countries += 1
    if len(self.countriesLeft) == 0:
      self.gamePhase = 'initialFortify'
    self.endTurn()
         
  def initialPickOneComputer(self):    
    
    p = self.activePlayer
    if p.human: return # only made for AI that has  pickCountry method
    p = self.activePlayer
    country = p.pickCountry()
    
    if not country is None and country.owner == -1:
      country.owner = p.code
      country.addArmies(1)
      self.countriesLeft.remove(country)
      p.initialArmies -= 1
      p.num_countries += 1
    self.endTurn()
    
        
  def initialFortifyHuman(self, country, armiesToAdd = 1):
       
    if isinstance(country, int):
      country = self.world.countries[country]
    p = self.activePlayer
    if not p.human: return # only made to wait human picks 
    if armiesToAdd<0: return # ERROR
    if not country is None and country.owner == p.code:
      #armiesToAdd = 0 means max possible
      #armiesToAdd > 0 means try that number, unless the income is less
      if armiesToAdd == 0:
        armies = p.income
      else:
        armies = min(p.income, armiesToAdd)
      country.addArmies(armies)
      p.income -= armies
      
    if p.income == 0: 
      self.endTurn()
  
  def initialFortifyComputer(self):    
    p = self.activePlayer
    armies = p.income    
    p.placeInitialArmies(armies)
    p.income -= armies    
    self.endTurn()
    
  def startTurnPlaceArmiesHuman(self, country, armiesToAdd = 0): 
    if isinstance(country, int):
      country = self.world.countries[country]
    p = self.activePlayer
    if not p.human: return # only made to wait human picks   
    if armiesToAdd<0: return # ERROR    
    if not country is None and country.owner == p.code:
      if armiesToAdd == 0:
        armies = p.income
      else:
        armies = min(p.income, armiesToAdd)
      
      country.addArmies(armies)
      p.income -= armies
      
    if p.income == 0: self.gamePhase = 'attack'
       
  # Methods to use in basic game turn ----------------------
  
  def updateIncome(self, p):    
    p.income = 0
    if self.gamePhase == "startTurn":
      for i, c in self.world.continents.items():
        if c.owner == p.code:
          p.income += int(c.bonus)    
      base = self.getNumberOfCountriesPlayer(p.code)
      p.income += max(int(base/3),3)
    elif self.gamePhase == "initialFortify":
      p.income = min(p.initialArmies, self.armiesPerTurnInitial)
      p.initialArmies -= p.income
  
  def setupNewRound(self):
    self.turnCount += 1
    if self.turnCount == 1: return
    # Update continent bonus if it is not the first round
    for i, cont in self.world.continents.items():
      cont.bonus = cont.bonus*(1+self.continentIncrease)
      
  def prepareStart(self):
    if self.gamePhase == 'startTurn' or self.gamePhase == 'initialFortify':
      if self.tookOverCountry:
        self.updateContinents()        
      self.updateIncome(self.activePlayer)
    self.tookOverCountry = False
  
  def endTurn(self):
  
    if self.tookOverCountry and self.useCards:
      self.activePlayer.cards.append(self.deck.draw())
    
    # Next player
    self.activePlayer = next(self.playerCycle)
    self.prepareStart()
  
  
  def updateMovable(self):    
    for c in self.countries:
      c.movable_armies = c.armies

  def updateContinents(self):
    for i, c in self.world.continents.items():
      c.updateOwner()
  '''
  Function that represents the turn of a player
  Designed to work with pygame
  '''
  def play(self):
    
    p = self.activePlayer
    if p.code == self.firstPlayerCode and self.gamePhase == "startTurn":
      # New round has begun
      self.setupNewRound()
    if not p.is_alive:
      self.endTurn()
      return 
      
    # Initial phase
    if 'initial' in self.gamePhase:
      if self.initialPhase:
        # Initial Pick
        if self.gamePhase == 'initialPick':
          if self.pickInitialCountries:
            # The picking will end with a endTurn(), so everything is fine
            if not p.human:
              self.initialPickOneComputer()
              if len(self.countriesLeft)==0:
                self.gamePhase = 'initialFortify'
            else: #Must wait for manual call to pickInitialOneHuman
              pass
          else:
            # No endTurn(), so we must do it manually here
            self.randomInitialPick()
            self.prepareStart()
        
        # Initial Fortify
        if self.gamePhase == 'initialFortify':
          over = False
          if p.initialArmies==0 and p.income==0:
            # Check if the phase is finished
            N = self.getNumberOfPlayers()
            cont = 1
            q = next(self.playerCycle)
            while (q.initialArmies==0 and q.income==0) and cont < N:
              cont += 1
              q = next(self.playerCycle)
            if cont >= N: 
              over=True
            else:
              self.activePlayer = q
            
          if not over:
            if not self.activePlayer.human:        
              self.initialFortifyComputer()          
            else:
              pass
          else:
            # Go to last player before endTurn()
            while self.activePlayer.code != self.lastPlayerCode:
              self.activePlayer = next(self.playerCycle)
            self.gamePhase = 'startTurn'
            self.endTurn()
      else:
        print("No initial phase, setting everything random")
        # No initial phase, everything at random
        print("random pick")
        self.randomInitialPick()
        self.prepareStart() # So that next player has income updated
        print("random fortify")
        self.randomInitialFotify()
        self.prepareStart() # So that next player has income updated
      

    # Start turn: Give armies and place them
    
    if self.gamePhase == 'startTurn':
      
      
      if not p.human:
        armies = p.income
        res = p.cardPhase(p.cards) if self.useCards else 0
        armies += res
        p.placeArmies(armies)
        p.income -= armies
        self.gamePhase = 'attack'
      else:
        pass
      
    # Attack
    if self.gamePhase == 'attack':
      if not p.human:
        p.attackPhase()     
        self.gamePhase = 'fortify'
        
    
    # Fortify
    if self.gamePhase == 'fortify':
      if not p.human:
        self.updateMovable()
        p.fortifyPhase()
        self.gamePhase = 'end'


    # End turn
    if self.gamePhase == 'end':          
      self.gamePhase = 'startTurn'
      self.endTurn()



  ''' Cashes in the given card set. Each parameter must be a reference  to a different Card instance sent via cardsPhase(). 
  It returns true if the set was cashed, false otherwise. '''
  
  def cashCards(self, card, card2, card3):
    # Check that cards are a set, take them out from player
    # Deal also with bonuses from country cards
    #   If card is from owned country, receive extra 2 armies there
    if Deck.isSet(card, card2, card3):
      res = self.nextCashArmies
      self.nextCashArmies = self.cardSequence.nextCashArmies()
      for c in [card, card2, card3]:
        if c.code < 0: continue
        if self.world.countries[c.code].owner == self.activePlayer.code:
          self.world.countries[c.code].addArmies(2)
        self.activePlayer.cards.remove(c)
      self.activePlayer.income += res
      return res
    else:
      return 0
  
  
  ''' Places numberOfArmies armies in the given country. '''
  def placeArmies(self, numberOfArmies:int, country):    
    code = country.code if isinstance(country, Country)  else country
    if self.world.countries[code].owner != self.activePlayer.code:
      return -1
    
    self.world.countries[code].addArmies(numberOfArmies)
    return 0
   
  
  def roll(self, attack:int = 3, defense:int = 2):
    aDice = sorted(np.random.randint(0,7,size=attack), reverse=True)
    dDice = sorted(np.random.randint(0,7,size=defense), reverse=True)
    aLoss, dLoss = 0, 0
    for a, d in zip(aDice, dDice):
      if a<=d:
        aLoss += 1
      else:
        dLoss += 1
    
    return aLoss, dLoss
  ''' 
  If *attackTillDead* is true then perform attacks until one side or the other has been defeated.
  Otherwise perform a single attack.
  This method may only be called from within an agent's attackPhase() method.
  The Board's attack() method returns symbolic ints, as follows:     
    - a negative return means that you supplied incorrect parameters.
    - 0 means that your single attack call has finished, with no one being totally defeated. Armies may have been lost from either country.
    - 7 means that the attacker has taken over the defender's country.
    NOTE: before returning 7, board will call moveArmiesIn() to poll you on how many armies to move into the taken over country.
    - 13 means that the defender has fought off the attacker (the attacking country has only 1 army left).
  '''
  def attack(self, countryCodeAttacker: int, countryCodeDefender:int, attackTillDead:bool) -> int:
    cA = self.world.countries[countryCodeAttacker]
    cD = self.world.countries[countryCodeDefender]
    attacker = self.players[cA.owner]
    defender = self.players[cD.owner]
    if attacker.code != self.activePlayer.code: return -1
    if defender.code == self.activePlayer.code: return -1
    stop = False
    while not stop:
      aLoss, dLoss = self.roll(min(3, cA.armies-1), min(2,cD.armies))
      cA.armies -= aLoss
      cD.armies -= dLoss
      if cD.armies < 1: 
        # Attacker won
        armies = attacker.moveArmiesIn(countryCodeAttacker, countryCodeDefender)
        cA.armies -= armies
        attacker.num_countries += 1
        defender.num_countries -= 1
        cD.owner = attacker.code
        cD.armies += armies
        
        if defender.num_countries == 0:
          defender.is_alive = False
          # Player has been eliminated. 
          # Must do cardPhase if more than 5 cards or self.immediateCash==True
          if self.useCards:
            if self.transferCards:
              attacker.cards.extend(defender.cards)
              if len(attacker.cards)>=5 or self.immediateCash:
                # Bot players - Run cardPhase
                if not attacker.human:
                  armies = attacker.cardPhase(attacker.cards)
                  if armies > 0:
                    attacker.placeArmies(armies)            
                else: # Human player
                  # leave them to be properly cashed at startTurn
                  pass
                  #if len(attacker.cards)>=5:
                  #  card_set = Deck.yieldCashableSet(attacker.cards)
                  #  self.cashCards(*card_set)
        self.tookOverCountry = True        
        return 7
      
      if cA.armies < 2:
        # Defender won
        return 13        
      if not attackTillDead: stop=True
    return 0
    
      
  '''
  Order a fortification move.  
  This method may only be called from within an agent's 'fortifyPhase()' method. It returns 1 on a successful fortify, 0 if no armies could be fortified (countries must always keep 1 army on them) and a negative number on failure.  
  '''
  def fortifyArmies(self, numberOfArmies:int, origin, destination) ->int:
    cO = origin if isinstance(origin, Country) else self.world.countries[origin]
    cD = destination if isinstance(destination, Country) else self.world.countries[destination]
    
    if cO.owner != self.activePlayer.code: return -1
    if cD.owner != self.activePlayer.code: return -1
    if cO.movable_armies < numberOfArmies: return -1
    # Even if movable_armies == armies, we have to always leave one army at least    
    aux = numberOfArmies-1 if numberOfArmies == cO.armies else numberOfArmies
    
    cO.armies -= aux
    cO.movable_armies -= aux
    cD.armies += aux
    return 0
    


  ### Info methods ------------------------------------------
  
  # These methods are provided for the agents to get information about the game.

  ''' Will return an array of all the countries in the game. The array is ordered by country code, so c[i].getCode() equals i.  '''
  def getCountries(self):
    return self.countries
  
  ''' Return a country by id'''
  def getCountryById(self, ID):
    for c in self.countries:
      if c.id == ID: return c
    return None
  
  
  ''' Returns the number of countries in the game.  '''
  def getNumberOfCountries(self)->int:
    return len(self.world.countries)
    
  ''' Returns the number of countries owned by player.  '''
  def getNumberOfCountriesPlayer(self, player:int)->int:
    s = 0
    for c in self.countries:
      if c.owner==player: s += 1
    return s
  
  
  ''' Returns the number of continents in the game.  '''  
  def getNumberOfContinents(self) -> int:
    return len(self.world.continents)

  ''' Returns the number of bonus armies given for owning the specified continent.  '''  
  def getContinentBonus(self, cont:int )->int:
    c = self.world.continents.get(cont)
    return c.bonus if not c is None else None
  

  ''' Returns the name of the specified continent (or null if the map did not give one).  '''  
  def getContinentName(self, cont:int) -> str:
    c = self.world.continents.get(cont)
    if not c is None:
      return c.name  
    else:
      return None
  
  
  ''' Returns the number of players that started in the game.  '''  
  def getNumberOfPlayers(self) -> int:
    return self.startingPlayers   
  
  ''' Returns the number of players that are still own at least one country.  '''  
  def getNumberOfPlayersLeft(self) -> int:
    return sum([p.is_alive for _, p in self.players.items()])
  
  ''' Returns the current income of the specified player.  '''  
  def getPlayerIncome(self, player:int) -> int:
    p = self.players.get(player)
    if not p is None:
      return p.income
    
  ''' Returns the TextField specified name of the given player.  '''  
  def getPlayerName(self, player:int) -> str:
    p = self.players.get(player)
    if not p is None:
      return p.name()

  ''' Returns whatever the name() method of the of the given agent returns.  '''
  def getAgentName(self, player:int) -> str:
    return self.getPlayerName(player)
  
  ''' Returns the number of cards that the specified player has.  '''  
  def getPlayerCards(self, player:int) -> int:
    p = self.players.get(player)
    if not p is None:
      return len(p.cards)
  
  
  ''' Returns the number of armies given by the next card cash.  '''  
  def getNextCardSetValue(self) -> int:
    return self.nextCashArmies

  ''' Returns true if the current player has taken over a country this turn. False otherwise. '''
  def tookOverACountry(self) -> bool:
    return self.tookOverCountry
  
  
  ''' Returns whether or not cards are on in the preferences.  '''
  def useCards(self) -> bool:
    return self.useCards
  

  ''' Returns whether or not cards get transferred when a player is killed.  '''
  def transferCards(self) -> bool:
    return self.transferCards
  
  ''' Returns whether or not cards are immediately cashed when taking over a player and ending up with 5 or more cards.  '''
  def immediateCash(self) -> bool:
    return self.immediateCash  



  ''' Return a string representation the card progression for this game. If cards are not being used then it will return "0". '''
  def getCardProgression(self) -> str:
    aux = self.cardSequence.__cls__()
    res = []
    for _ in range(10):
      res.append(aux.nextCashArmies())
    return ' '.join(res)
  
  
  ''' Return the percent increase of the continents. '''
  def getContinentIncrease(self) -> int:
    return self.continentIncrease
    
  
  ''' Return the number of seconds left in this turn. '''
  def getTurnSecondsLeft(self) -> int:
    return 0
    
    
  ''' Return the count of the turn rounds for the game '''
  def getTurnCount(self) -> int:
    return self.turnCount
  
  
  #### From BoardHelper - Seems logic to have them here with access to the board

  def getPlayersBiggestArmy(self, player:int)-> Country:
    m, biggest = -1, None    
    for c in self.countries:
      if c.owner == player and c.armies > m:
        m = c.armies
        biggest = c
    return biggest
  
  def getPlayersBiggestArmyWithEnemyNeighbor(self, player:int)-> Country:
    m, biggest = -1, None    
    for c in self.countries:
      if c.owner == player and c.armies > m:
        m = c.armies
        biggest = c
    return biggest
  
  def getPlayerArmies(self, player:int)-> int:
    s = 0
    for c in self.countries:
      if c.owner==player: s += c.armies
    return s
  
  def getPlayerCountries(self, player:int)-> int:
    s = 0
    for c in self.countries:
      if c.owner==player: s += 1
    return s
  
  def getPlayerArmiesInContinent(self, player:int, cont:int)-> int:
    s = 0
    for c in self.world.continents[cont].countries:
      if c.owner==player: s += c.armies
    return s
  
  
  def getEnemyArmiesInContinent(self, player:int, cont:int)-> int:
    s = 0
    for c in self.world.continents[cont].countries:
      if c.owner!=player: s += c.armies
    return s
  
  def getContinentBorders(self, cont:int)-> int:    
    border = []
    for c in self.world.continents[cont].countries:
      for o in c.successors():
        if o.continent != cont:
          border.append(c)
          break            
    return list(set(border))
  
  
  def getPlayerArmiesAdjoiningContinent(self, player:int, cont:int)-> int:
    s = 0
    border = self.getContinentBorders(cont)
    toCount = []
    for c in border:
      for o in c.predecessors():
        if o.owner == player:
          toCount.append(o)
    
    # Remove duplicates
    toCount = set(toCount)
    for c in toCount:
      s += c.armies
    return s
  
  def playerIsStillInTheGame(self, player:int):
    p = self.players.get[player]
    if not p is None: return p.is_alive
    return None
    
  def numberOfContinents(self):
    return len(self.world.continents)
    
  def getContinentSize(self, cont:int):
    return len(self.world.continents[cont].countries)
  
  def getCountryInContinent(self, cont:int):
    countries = self.world.continents[cont].countries
    return np.random.choice(countries).code
    
  def getContinentBordersBeyond(self, cont:int)-> int:    
    border = []
    for c in self.world.continents[cont].countries:
      for o in c.successors():
        if o.continent != cont:
          border.append(o)
    return list(set(border))
   
  def playerOwnsContinent(self, player:int, cont:int):
    return self.world.continents[cont].owner == player      
  
  def playerOwnsAnyContinent(self, player):
    for cont in self.world.continents:
      if cont.owner == player: return True
    return False

  def playerOwnsAnyPositiveContinent(self, player):
    for cont in self.world.continents:
      if cont.bonus>0 and cont.owner == player: return True
    return False
    
  def anyPlayerOwnsContinent(self, cont:int):    
    return self.world.continents[cont].owner != -1
    
  def playerOwnsContinentCountry(self, player, cont):
    for c in self.world.continents[cont].countries:
      if c.owner == player: return True
    return False

  def getSmallestEmptyCont(self):
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if c.isEmpty() and len(c.countries)<size:
        smallest = c
        size = len(c.countries)
    return smallest
  
  def getSmallestPositiveEmptyCont(self):
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if c.isEmpty() and len(c.countries)<size and c.bonus>0:
        smallest = c
        size = len(c.countries)
    return smallest
    
  def getSmallestOpenCont(self):
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if c.isOpen() and len(c.countries)<size:
        smallest = c
        size = len(c.countries)
    return smallest
    
  def getSmallestPositiveOpenCont(self):
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if c.isOpen() and len(c.countries)<size and c.bonus>0:
        smallest = c
        size = len(c.countries)
    return smallest
    
  def closestCountryWithOwner(self, country:int, owner:int):
    G = self.world.map_graph
    paths = single_source_shortest_path(G, country, cutoff=G.num_nodes)
    m = sys.maxsize
    closest = None
    for i, le in paths.items():
      if self.world.countries[i].owner == owner:
        if m > le:
          m = le
          closest = self.world.countries[i]
    return closest
  
  def easyCostCountryWithOwner(self, country:int, owner:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    '''
    pass
    
  def easyCostFromCountryToContinent(self, country:int, cont:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    '''
    pass
   
  def easyCostBetweenCountries(self, source:int, target:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    '''
    pass
    
  def friendlyPathBetweenCountries(self, source:int, target:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    '''
    pass
    
  def cheapestRouteFromOwnerToCont(self, owner:int, cont:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    '''
    pass
  
  def getCountriesCopy(self):
    return copy.deepcopy(self.countries)

  def getAttackListTarget(self, target:int):
    '''
    Countries that can attack the target
    '''
    return list(self.world.countries[target].predecessors())
    
  def getAttackListSource(self, target:int):
    '''
    Countries that can attack the target
    '''
    return list(self.world.countries[target].successors())
    
  def getCountriesPlayer(self, player:int):    
    return [c for c in self.countries if c.owner==player]
    
  
  ''' Gives a String representation of the board. '''
  def __repr__(self):
    print('Board with {} players'.format(self.startingPlayers))  
   
  
 

#%% TESTING
if __name__ == '__main__':

  console_debug = True
  
  # Load map
  mapLoader = MapLoader('../support/maps/classic_world_map.json')
  mapLoader.load_from_json()
  world = World(mapLoader)
  
  
  
  # Set players
  
  p1, p2, p3, p4 = agent.Human('PocketNavy'), agent.RandomAgent('Blue'), agent.RandomAgent('Green'), agent.RandomAgent('Yellow')
  #p1.console_debug = console_debug  
  players = [p1, p2, p3, p4]
  # Set board
  board = Board(world, players)
  board.pickInitialCountries = False
  

  
    
  
