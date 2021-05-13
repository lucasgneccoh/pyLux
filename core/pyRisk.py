import networkx as nx
from networkx.readwrite import json_graph
import json
import sys
import itertools
import numpy as np
import copy
import random
import time
import pandas as pd

from pyLux.core import agent
from pyLux.core.deck import Deck, ListThenArithmeticCardSequence
from pyLux.core.world import World, Country, Continent


from string import ascii_lowercase

def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


#%% Board
class Board(object):
  '''! Class containing all the information about the world and about the 
  players. It is used to play the actual game, and contains the game logic 
  and flow.
  '''
  board_codes = iter_all_strings()
  
  def __init__(self, world, players):
    '''! Setup the board with the given world representing the map, 
    and the list of players. We take different game settings either from 
    the world or from a dictionary of preferences to allow players to 
    easily customize their games
    '''
    
    self.board_id = next(Board.board_codes)
  
    # Game setup
    self.world = world
    N = len(players)
    
    # TODO: This should come from the MapLoader
    # Because it will change in each map
    armiesForPlayers = {2:45, 3: 35, 4: 30, 5: 25, 6:20}
    if N > 6: 
      initialArmies = 20
    elif N < 2:
      raise Exception("Minimum 2 players to play pyRisk")
    else:
      initialArmies = armiesForPlayers[N]
    
    # random.shuffle(players) Make this outside for now
    self.orig_players = copy.deepcopy(players)
    self.players = {i: p for i, p in enumerate(players)}
    for i, p in self.players.items():
      p.setPrefs(i)
      
    self.startingPlayers = N    
    self.nextCashArmies = 0 
    self.tookOverCountry = False
    self.roundCount = 0    
    
    
    self.playerCycle = itertools.cycle(list(self.players.values()))
    self.activePlayer = next(self.playerCycle)
    self.firstPlayerCode = self.activePlayer.code
    self.lastPlayerCode = list(self.players.values())[-1].code
    self.cacheArmies = 0
    self.gamePhase = 'initialPick'
    self.gameOver = False
     
    # Preferences (Default here)
    self.prefs = {}
    self.initialPhase = True
    self.useCards = True
    self.transferCards = True
    self.immediateCash = False
    self.turnSeconds = 30
    self.continentIncrease = 0.05
    self.pickInitialCountries = False
    self.armiesPerTurnInitial = 4
    self.console_debug = False
    
    # Fixed for the moment (must be copied manually)
    self.cardSequence = ListThenArithmeticCardSequence(sequence=[4,6,8,10,12,15], incr=5)
    self.deck = Deck()
    num_wildcards = len(self.world.countries)//20
    self.deck.create_deck(self.countries(), num_wildcards=num_wildcards)
    self.nextCashArmies = self.cardSequence.nextCashArmies()
    self.aux_cont = 0
    
    # Start players
    for _, p in self.players.items():
      p.is_alive = True
      p.cards = []
      p.income = 0
      p.initialArmies = int(initialArmies)
      p.num_countries = 0

  
    
  @staticmethod
  def fromDicts(continents:dict, countries:dict, inLinks:dict,\
                players:dict, misc:dict):
    world = World.fromDicts(continents, countries, inLinks)
    new_players = {}
    for i, attrs in players.items():
      # TODO: Maybe define a way to change this default agent
      p = agent.RandomAgent(attrs['name'])
      new_players[i] = p
    board = Board(world, list(new_players.values()))
    prefs = misc.get('prefs')
    if not prefs is None:
      board.setPreferences(prefs)
      
    # TODO: give players theirs cards (in number)
    for i, attrs in players.items():
      for n in range(int(attrs['cards'])):
        new_players[i].cards.append(board.deck.draw())
    board.gamePhase = misc['gamePhase']
    while board.activePlayer.code != int(misc['activePlayer']):
      board.activePlayer = next(board.playerCycle)
    return board
  
  def toDicts(self):
    ''' Used to transform the board into a simple representation like the one given by Java
      when using a Python player in java.
      The idea is that this representation can be the input of general players, and also
      a more compact way of giving the board to a neural network
      
      Returns:
        Dict of countries ('code', 'name', 'continent', 'owner', 'armies', 'moveableArmies')
        Dict of continents ('code', 'name', 'bonus')
        Dict with player information ('code', 'name', 'income', 'cards')
        Dict with incoming links for each node
        Dict with other information, such as the gamePhase, nextCashArmies, etc
      
    '''
    
    countries = {c['code']: c for c in self.countries()}
    continents = {i: c for i,c in self.world.continents.items()}
    players = {p.code:{'code':p.code, 'name':p.name(),
                       'income': self.getPlayerIncome(p.code),
                       'cards':len(p.cards)} for i, p in self.players.items()}
    inLinks = {n.code: [c.code for c in self.world.predecessors(n.code)] for n in self.countries()}
    misc = {'gamePhase': self.gamePhase, 'activePlayer':self.activePlayer.code}
    return continents, countries, inLinks, players, misc
    
  
  def setPreferences(self, prefs):
    self.prefs = prefs
    available_prefs = ['initialPhase', 'useCards', 'transferCards', 'immediateCash', 'continentIncrease', 'pickInitialCountries', 'armiesPerTurnInitial', 'console_debug']
    # In the future, card sequences, num_wildcards, armiesInitial
    for a in available_prefs:
      r = prefs.get(a)
      if r is None:
        print(f"Board preferences: value for '{a}' not given. Leaving default value {getattr(self, a)}")
      else:
        setattr(self, a, r)
        
#%% Board: Play related functions
  
  def randomInitialPick(self):
    '''! Picks countries at random for all the players until no empty country is left
    '''
    
    player_codes = itertools.cycle(list(self.players.keys()))
    countriesLeft = self.countriesLeft()
    random.shuffle(countriesLeft)
    while countriesLeft:
      c = countriesLeft.pop()
      p = next(player_codes)
      c.owner = p
      c.addArmies(1)
      self.players[p].initialArmies -= 1
      self.players[p].num_countries += 1  
      if self.console_debug: print(f"Board:randomInitialPick: Picked {c.id} for {p} ({self.players[p].name()})")
    self.gamePhase = 'initialFortify'
    
  def randomInitialFotify(self):
    '''! Performs the initial fortify phase at random, putting armies for all the players until no more initial armies are left for any player.
    '''
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
    '''! For the GUI play only: Function added to simplify the GUI code. It updates the board by picking the chosen country for the human player
    '''
    if isinstance(country, int):
      country = self.world.countries[country]
    p = self.activePlayer
    if not p.human: return # only made to wait human picks      
    if not country is None and country.owner == -1:
      country.owner = p.code
      country.addArmies(1)      
      p.initialArmies -= 1
      p.num_countries += 1
    if len(self.countriesLeft()) == 0:
      self.gamePhase = 'initialFortify'
    self.endTurn()
         
  def initialPickOneComputer(self):
    '''! Calls the agent.pickCountry() method so that the bots can pick their countries
    '''   
    p = self.activePlayer
    if p.human: return # only made for AI that has  pickCountry method
    
    country = p.pickCountry(self)
    
    if not country is None and country.owner == -1:
      country.owner = p.code
      country.addArmies(1)            
      p.initialArmies -= 1
      p.num_countries += 1
      if self.console_debug:
        print(f'initialPickOneComputer:Picked {country.id}')
      
    self.endTurn()
    
        
  def initialFortifyHuman(self, country, armiesToAdd = 1):
    '''! For the GUI play only: Function added to simplify the GUI code. It updates the board by putting the initial armies in the chosen country for the human player during the initial fortify game phase
    '''
       
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
    '''! Calls the agent.placeInitialArmies() method so that the bots can fortify their countries in the initial fortify phase
    '''
    
    p = self.activePlayer
    armies = p.income    
    p.placeInitialArmies(self, armies)
    p.income -= armies    
    self.endTurn()
    
  def startTurnPlaceArmiesHuman(self, country, armiesToAdd = 0): 
    '''! For the GUI play only: Function added to simplify the GUI code. It updates the board by putting the start turn armies in the chosen country for the human player during the start turn game phase
    '''
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
    '''! Calculates the income of the player p. This function should only be called during the "startTurn" or "initialFortify" game phases. In any other case, the income will be set to 0.
    '''    
    if not p.is_alive:
      p.income = 0
      return 
    
    if self.gamePhase == "startTurn":
      p.income = 0
      for i, c in self.world.continents.items():
        if c.owner == p.code:
          p.income += int(c.bonus)    
      base = self.getNumberOfCountriesPlayer(p.code)
      p.income += max(int(base/3),3)
    elif self.gamePhase == "initialFortify":
      if p.income >0:
        # There are armies still in income
        pass
      else:
        p.income = min(p.initialArmies, self.armiesPerTurnInitial)
        p.initialArmies -= p.income
  
  def setupNewRound(self):
    '''! Performs the necessary actions at the end of a round (all players played)
    '''
    self.roundCount += 1
    if self.roundCount == 1: return
    
    # Update continent bonus if it is not the first round
    for i, cont in self.world.continents.items():
      cont.bonus = cont.bonus*(1+self.continentIncrease)
    
    s = 0
    for i, p in self.players.items():
      if p.is_alive: s += 1
    
    if s == 1: self.gameOver = True
    if self.console_debug: print(f"Board:setupNewRound: Found {s} players alive. gameOver = {self.gameOver}")
    return
      
  def prepareStart(self):
    '''! At the very beginning of a player's turn, this function should be called to update continent ownership and active player income.
    '''
    if self.gamePhase == 'startTurn' or self.gamePhase == 'initialFortify':
      if self.tookOverCountry:
        self.updateContinents()        
      self.updateIncome(self.activePlayer)
    self.tookOverCountry = False
  
  def endTurn(self):
    '''! Function to end a player's turn by giving it a card if at least one country was conquered, change the active player, and prepare his turn by calling self.prepareStart()
    '''
  
    if self.tookOverCountry and self.useCards:
      self.activePlayer.cards.append(self.deck.draw())
    
    # Next player
    self.activePlayer = next(self.playerCycle)
    if self.console_debug: print(f"Board:endTurn: Ending turn: next player is ({self.activePlayer.code}) {self.activePlayer.name()}")
    self.prepareStart()
  
  
  def updatemoveable(self):
    '''! Sets the moveable armies of every country equal to the number of armies. Should be called after every "attack" phase and before "fortify" phase
    '''
    for c in self.countries():
      c.moveableArmies = c.armies

  def updateContinentOwner(self, cont):
    '''! Looks over the member countries to see if the same player owns them all
    '''
    continent = self.world.continents[cont]
    c_code = continent.countries[0]
    p = self.world.countries[c_code].owner
    for c in continent.countries:
      if p != self.world.countries[c].owner:
        continent.owner = -1
        return
        
    continent.owner = p
    
  def updateContinents(self):
    '''! Updates continent ownership for every continent
    '''
    for i, _ in self.world.continents.items():
      self.updateContinentOwner(i)
  
  def play(self):
    '''! Function that represents the turn of a player
    Designed to work with pygame.
    This function will call the gameplay function sfrom the agents
    '''
    if self.gameOver:
      if self.console_debug: print("Board:play: Game over")
      return
    
    
    p = self.activePlayer
    if self.console_debug: print(f"Board:play: Starting play for ({p.code}), {p.name()}")
    
    if not p.is_alive:
      if self.console_debug: print("Board:play: Returning, player is not alive")
      self.endTurn()

    else:
    
      #if self.console_debug: print(f"Board {self.board_id}:play: ({p.code}) {p.name()}: {self.gamePhase}")
      # Initial phase
      if 'initial' in self.gamePhase:
        if self.initialPhase:
          # Initial Pick
          if self.gamePhase == 'initialPick':
            if self.pickInitialCountries:
              # The picking will end with a endTurn(), so everything is fine
              if not p.human:
                #print("Board:play: Launching initialPickOneComputer")
                self.initialPickOneComputer()
                if len(self.countriesLeft())==0:
                  self.gamePhase = 'initialFortify'
                  return
              else: #Must wait for manual call to pickInitialOneHuman
                #print("Board:play: Initial pick huamn. Waiting")
                pass
            else:
              # No endTurn(), so we must do it manually here
              self.randomInitialPick()
              self.prepareStart()
              return
          
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
              return
        else:
          if self.console_debug: print("No initial phase, setting everything random")
          # No initial phase, everything at random
          if self.console_debug: print("random pick")
          self.randomInitialPick()
          self.prepareStart() # So that next player has income updated
          if self.console_debug: print("random fortify")
          self.randomInitialFotify()
          # Go to last player before endTurn()
          while self.activePlayer.code != self.lastPlayerCode:
            self.activePlayer = next(self.playerCycle)
          self.gamePhase = 'startTurn'
          self.endTurn() # So that next player has income updated
          return
        

      # Start turn: Give armies and place them    
      try:
        if self.console_debug: print(f"Board:play: {self.gamePhase}")
        if self.gamePhase == 'startTurn':      
          if not p.human:
            armies = p.income
            cashed = 0
            if self.useCards:
              # Card cash          
              card_set = p.cardPhase(self, p.cards)
              if not card_set is None:
                cashed += self.cashCards(*card_set)
    
              # If player has more than 5 cards, keep asking
              # If player does not cash, must force the cash
              while len(p.cards)>=5:
                card_set = p.cardPhase(self, p.cards)
                if not card_set is None:
                  cashed += self.cashCards(*card_set)
                else:
                  # Force the cash
                  card_set = Deck.yieldCashableSet(p.cards)
                  if not card_set is None:
                    cashed += self.cashCards(*card_set)
                  else:
                    # Error
                    print("More than 5 cards but not able to find a cashable set. Maybe the deck has too many wildcards?")
                    break
              armies += cashed
            
    
            p.placeArmies(self, armies)
            p.income = 0
            self.gamePhase = 'attack'
          else:
            pass
      except Exception as e:
        print(e)
        raise(e)
        
      # Attack
      if self.gamePhase == 'attack':
        if self.console_debug: print(f"Board:play: {self.gamePhase}")
        if not p.human:
          p.attackPhase(self)     
          self.gamePhase = 'fortify'
          
      
      # Fortify
      if self.gamePhase == 'fortify':
        if self.console_debug: print(f"Board:play: {self.gamePhase}")
        if not p.human:
          self.updatemoveable()
          p.fortifyPhase(self)
          self.gamePhase = 'end'


      # End turn
      if self.gamePhase == 'end':
        if self.console_debug: print(f"Board:play: {self.gamePhase}")        
        self.gamePhase = 'startTurn'
        self.endTurn()
        if self.console_debug: print("Board:play: Returning, end of turn")

    if p.code == self.firstPlayerCode and self.gamePhase == "startTurn":
      # New round has begun
      if self.console_debug: print("Board:play: Setting up new round")
      self.setupNewRound()
      self.prepareStart()


  def cashCards(self, card, card2, card3):
    '''! Cashes in the given card set. Each parameter must be a reference to a different Card instance sent via cardsPhase(). 
    It returns the number of cashed armies'''
    if Deck.isSet(card, card2, card3):
      res = self.nextCashArmies
      self.nextCashArmies = self.cardSequence.nextCashArmies()
      if self.console_debug:
        print(f'Board:cashCards:Obtained {res} armies')
      for c in [card, card2, card3]:
        if c.code < 0: continue
        country = self.world.countries[c.code]
        if country.owner == self.activePlayer.code:
          country.addArmies(2)
          if self.console_debug:
            print(f'Board:cashCards:Bonus cards in {country.id}')
        try:
          self.activePlayer.cards.remove(c)
        except Exception as e:
          print(f'Board:cashCards: Card {c} does not belong to activePlayer {self.activePlayer.code}, {self.activePlayer.name()}: {self.activePlayer.cards}')
          raise e      
      return res
    else:
      return 0
  
  
  def placeArmies(self, numberOfArmies:int, country):
    '''! Places numberOfArmies armies in the given country. '''
    code = country.code if isinstance(country, Country)  else country
    if self.world.countries[code].owner != self.activePlayer.code:
      return -1    
    
    self.world.countries[code].addArmies(numberOfArmies)
    if self.console_debug:
      print(f'placeArmies:Placed {numberOfArmies} armies in {country.id}')
    return 0
   
  
  def roll(self, attack:int, defense:int):
    '''! Simulate the dice rolling. Inputs determine the number of dice to use for each side.
    They must be at most 3 for the attacker and 2 for the defender.
    Returns the number of lost armies for each side
    '''
    if attack > 3 or attack <= 0: return None
    if defense > 2 or defense <= 0: return None
    aDice = sorted(np.random.randint(0,7,size=attack), reverse=True)
    dDice = sorted(np.random.randint(0,7,size=defense), reverse=True)
    aLoss, dLoss = 0, 0
    for a, d in zip(aDice, dDice):
      if a<=d:
        aLoss += 1
      else:
        dLoss += 1
    
    return aLoss, dLoss
    
  def attack(self, countryCodeAttacker: int, countryCodeDefender:int, attackTillDead:bool) -> int:
    '''! Performs an attack.
    If *attackTillDead* is true then perform attacks until one side or the other has been defeated, otherwise perform a single attack.
    This method may only be called from within an agent's attackPhase() method.
    The Board's attack() method returns symbolic ints, as follows:     
      - a negative return means that you supplied incorrect parameters.
      - 0 means that your single attack call has finished, with no one being totally defeated. Armies may have been lost from either country.
      - 7 means that the attacker has taken over the defender's country.
      NOTE: before returning 7, board will call moveArmiesIn() to poll you on how many armies to move into the taken over country.
      - 13 means that the defender has fought off the attacker (the attacking country has only 1 army left).
    '''
    cA = self.world.countries[countryCodeAttacker]
    cD = self.world.countries[countryCodeDefender]
    attacker = self.players[cA.owner]
    defender = self.players[cD.owner]
    if attacker.code != self.activePlayer.code: return -1
    if defender.code == self.activePlayer.code: return -1
    if not cD in self.world.getAdjoiningList(cA.code, kind=1): return -1
    if cA.armies <= 1 or cD.armies == 0: return -1
    
    stop = False
    if self.console_debug:
      print(f'attack:From {cA.id} to {cD.id}, tillDead:{attackTillDead}')
    while not stop:
      aDice, dDice = min(3, cA.armies-1), min(2,cD.armies)      
      aLoss, dLoss = self.roll(aDice, dDice) 
      if self.console_debug:
        print(f'attack:Dice roll - attacker loss {aLoss},  defender loss {dLoss}')
      cA.armies -= aLoss
      cD.armies -= dLoss
      if cD.armies < 1: 
        stop=True
        # Attacker won
        armies = attacker.moveArmiesIn(self, countryCodeAttacker, countryCodeDefender)
        if armies >= cA.armies: armies = cA.armies-1
        if armies < aDice: armies = aDice
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
              armies = 0
              while len(attacker.cards)>=5 and self.immediateCash:
                # Bot players - Run cardPhase                
                if not attacker.human:
                  card_set = attacker.cardPhase(self, attacker.cards)
                  if not card_set is None:
                    if self.console_debug:
                      print("****************************")
                      print("card set", card_set)
                      print(f"Trading cards for {self.activePlayer.code}, {self.activePlayer.name()}")
                      print(f"ActivePlayer: {self.activePlayer}")
                      print(f"ActivePlayer cards: {self.activePlayer.cards}")
                      print(f"Attacker: {attacker.code}, {attacker.name()}")
                      print(f"Attacker: {attacker}")
                      print(f"Attacker cards: {attacker.cards}")
                      print("****************************")
                    armies += self.cashCards(*card_set)
                  else:
                    if len(attacker.cards)>=5:
                      # Must force the cash
                      card_set = Deck.yieldCashableSet(attacker.cards)
                      if card_set is None:
                        # This should not arrive
                        if self.console_debug:
                          print('Board:attack: Error on forced cash')
                        pass
                      else:
                        armies += self.cashCards(*card_set)                  
                else: # Human player
                  # leave them to be properly cashed at startTurn
                  break
              if armies > 0:
                attacker.placeArmies(self, armies)
                 
        self.tookOverCountry = True    
        if self.console_debug:
          print('attack:Attack end: res 7')
        return 7
      
      if cA.armies < 2:
        # Defender won
        if self.console_debug:
          print('attack:Attack end: res 13')
        return 13        
      if not attackTillDead: stop=True
    if self.console_debug:
          print('attack:Attack end: res 0')
    return 0
    
      
  def fortifyArmies(self, numberOfArmies:int, origin, destination) ->int:
    '''! Order a fortification move. This method may only be called from within an agent's 'fortifyPhase()' method. It returns 1 on a successful fortify (maybe nothing happened), -1 on failure.
  '''
    cO = origin if isinstance(origin, Country) else self.world.countries[origin]
    cD = destination if isinstance(destination, Country) else self.world.countries[destination]
 
    if cO.owner != self.activePlayer.code: return -1
    if cD.owner != self.activePlayer.code: return -1
    if not cD in self.world.getAdjoiningList(cO.code, kind=1): return -1
    if cO.moveableArmies < numberOfArmies: return -1
 
    # Even if moveableArmies == armies, we have to always leave one army at least    
    aux = numberOfArmies-1 if numberOfArmies == cO.armies else numberOfArmies
    
    if self.console_debug:
      print(f'Board:Fortify: From {cO.id} to {cD.id}, armies = {aux}')
    cO.armies -= aux
    cO.moveableArmies -= aux
    cD.armies += aux
    return 1
    
  # Methods to "hard-play". That means playing without the "safe" interface of the play function. Useful for simulations
  def outsidePickCountry(self, country_code:int):
    c = self.world.countries.get(country_code)
    if c is None: return -1
    if c.owner != -1: return -1
    
    p = self.activePlayer    
    # print(f'Player is {p.code} {p.name()}')
    # print(f'Chosen country is {c.id} {c.name}')  
    if self.console_debug:
      print(f'outsidePickCountry: Picked {c.id}')
    c.owner = p.code
    c.armies += 1    
    p.initialArmies -= 1
    p.num_countries += 1
    return 0
  
  def outsidePlaceArmies(self, country_code:int, armies:int):
    p = self.activePlayer
    c = self.world.countries.get(country_code)
    if c is None: return -1
    if c.owner != p.code: return -1    
    if armies > p.income: return -1
    c.armies += armies
    p.income -= armies
    if self.console_debug:
      print(f'outsidePlaceArmies: Placed {armies} armies in {c.id}')
    if p.income == 0: self.endTurn()
    return 0
    
    
  
#%% Basic information methods
  # These methods are provided for the agents to get information about the game.

  def countries(self):
    '''! Will return a list of all the countries in the game. No order or guaranteed for now'''
    return [c for i, c in self.world.countries.items()]
    
  def countriesLeft(self):
    '''! Will return a list of all the countries without an owner'''
    return [c for c in self.countries() if c.owner ==-1]
  
  
  def getCountryById(self, ID):
    '''! Return a country by id or None if no match was found'''
    for c in self.countries():
      if c.id == ID: return c
    return None
  
  def getNumberOfCountries(self)->int:
    '''1 Returns the number of countries in the game.'''
    return len(self.world.countries)
    
  def getNumberOfCountriesPlayer(self, player:int)->int:
    '''! Returns the number of countries owned by a player.'''
    r = self.players.get(player)
    return r.num_countries if not r is None else None
  
  def getNumberOfContinents(self) -> int:
    '''! Returns the number of continents in the game.  '''  
    return len(self.world.continents)

  def getContinentBonus(self, cont:int )->int:
    '''! Returns the number of bonus armies given for owning the specified continent.  '''  
    c = self.world.continents.get(cont)
    return c.bonus if not c is None else None
    
  def getContinentName(self, cont:int) -> str:
    '''! Returns the name of the specified continent (or None if the map did not give one).  '''
    c = self.world.continents.get(cont)    
    return c.name if not c is None else None
     
  def getNumberOfPlayers(self) -> int:
    '''! Returns the number of players that started in the game.  ''' 
    return self.startingPlayers   
    
  def getNumberOfPlayersLeft(self) -> int:
    '''! Returns the number of players that are still own at least one country.  '''  
    return sum([p.is_alive for _, p in self.players.items()])
    
  def getPlayerIncome(self, player:int) -> int:
    '''! Returns the current income of the specified player.  '''  
    p = self.players.get(player)
    if not p is None:
      s = 0
      for i, c in self.world.continents.items():
        self.updateContinentOwner(i)
        if self.world.continents[i].owner == p.code:
          s += int(c.bonus)    
      base = self.getNumberOfCountriesPlayer(p.code)
      s += max(int(base/3),3)
      return s
      
  def getPlayerName(self, player:int) -> str:
    '''! Returns the TextField specified name of the given player.  '''  
    p = self.players.get(player)
    if not p is None:
      return p.name()
  
  def getAgentName(self, player:int) -> str:
    '''! Returns whatever the name() method of the of the given agent returns.  '''
    return self.getPlayerName(player)
  
  def getPlayerCards(self, player:int) -> int:
    '''! Returns the number of cards that the specified player has.  ''' 
    p = self.players.get(player)
    if not p is None:
      return len(p.cards)
        
  def getNextCardSetValue(self) -> int:
    '''! Returns the number of armies given by the next card cash. '''  
    return self.nextCashArmies


  def getCardProgression(self) -> str:
    '''! Return a string representation the card progression for this game. If cards are not being used then it will return "0". '''  
    aux = self.cardSequence.__cls__()
    res = []
    for _ in range(10):
      res.append(aux.nextCashArmies())
    return '-'.join(res)
  
#%% BoardHelper functions 
  #### From BoardHelper - Seems logic to have them here with access to the board
  # NOTE: As I will not be using these, they are not thoroughly tested for the moment

  def getPlayersBiggestArmy(self, player:int)-> Country:
    '''! Return the country belonging to player with the most armies'''
    m, biggest = -1, None    
    for c in self.countries():
      if c.owner == player and c.armies > m:
        m = c.armies
        biggest = c
    return biggest
  
  def getPlayersBiggestArmyWithEnemyNeighbor(self, player:int)-> Country:
    '''! Return the country belonging to player with the most armies and that may attack at least one country
    NOT FINISHED
    '''
    m, biggest = -1, None    
    for c in self.countries():
      if c.owner == player and c.armies > m:
        # Not finished
        m = c.armies
        biggest = c
    return biggest
  
  def getPlayerArmies(self, player:int)-> int:
    '''! Return the total number of armies that a player has on the board
    '''
    s = 0
    for c in self.countries():
      if c.owner==player: s += c.armies
    return s
  
  def getPlayerCountries(self, player:int)-> int:
    '''! Return the number of countries a player owns
    '''    
    return self.players[player].num_countries
  
  def getPlayerArmiesInContinent(self, player:int, cont:int)-> int:
    '''! Return the number of armies player has in the given continent
    '''    
    s = 0
    for i in self.world.continents[cont].countries:
      c = self.world.countries[i]
      if c.owner==player: s += c.armies
    return s
  
  
  def getEnemyArmiesInContinent(self, player:int, cont:int)-> int:
    '''! Return the number of enemy armies in the given continent. Eney means that is not player
    '''    
    s = 0
    for i in self.world.continents[cont].countries:
      c = self.world.countries[i]
      if c.owner!=player: s += c.armies
    return s
  
  def getContinentBorders(self, cont:int, kind=0)-> int:    
    '''! Return the list of countries in the given continent that are linked to another country outside of the continent. The link type is determined by the kind argument (See Country.getAdjoiningList)
    '''
    border = []
    for i in self.world.continents[cont].countries:
      c = self.world.countries[i]
      for o in self.world.getAdjoiningList(i, kind=kind):
        if o.continent != cont:
          border.append(c)
          break            
    return list(set(border))
  
  
  def getPlayerArmiesAdjoiningContinent(self, player:int, cont:int, kind = 0)-> int:
    '''! Return the total number of armies that player has around the continent. Use the kind parameter to look for armies that can attack the border of the continent, that can defend from an attack from the border, or both. 
    '''
    s = 0
    border = self.getContinentBorders(cont, kind=kind)
    toCount = []
    for c in border:
      for o in self.world.getAdjoiningList(c.code, kind=kind):
        if o.owner == player:
          toCount.append(o)
    
    # Remove duplicates
    toCount = set(toCount)
    for c in toCount:
      s += c.armies
    return s
  
  def playerIsStillInTheGame(self, player:int):
    '''! Return if player is alive or not
    '''
    p = self.players.get[player]
    if not p is None: return p.is_alive
    return None
    
  def numberOfContinents(self):
    '''! Return the number of continents in the world
    '''
    return len(self.world.continents)
    
  def getContinentSize(self, cont:int):
    '''! Return the number of countries in continent
    '''
    return len(self.world.continents[cont].countries)
  
  def getCountryInContinent(self, cont:int):
    '''! Return a random country code from a given continent
    '''
    countries = self.world.continents[cont].countries
    return np.random.choice(countries)
    
  def getContinentBordersBeyond(self, cont:int)-> int:
    pass
   
  def playerOwnsContinent(self, player:int, cont:int):
    '''! Return if player owns the whole continent
    '''
    self.updateContinentOwner(cont)
    return self.world.continents[cont].owner == player      
  
  def playerOwnsAnyContinent(self, player):
    '''! Return if player owns any continent
    '''
    for i in self.world.continents.items():
      self.updateContinentOwner(i)
      if self.world.continents[i].owner == player: return True
    return False

  def playerOwnsAnyPositiveContinent(self, player):
    '''! Return if player owns any continent that has positive bonus
    '''
    for i, _ in self.world.continents.items():
      self.updateContinentOwner(i)
      cont = self.world.continents[i]
      if cont.bonus>0 and cont.owner == player:
        return True
    return False
    
  def anyPlayerOwnsContinent(self, cont:int):
    '''! Return if continent is owned by any player
    '''
    self.updateContinentOwner(cont)
    return self.world.continents[cont].owner != -1
    
  def playerOwnsContinentCountry(self, player, cont):
    '''! Return if player owns at least one country in the given continent
    '''
    for i in self.world.continents[cont].countries:
      c = self.world.countries[i]
      if c.owner == player: return True
    return False

  def getSmallestEmptyCont(self):
    '''! Return the smallest continent that is completely empty
    '''
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if self.world.isEmptyContinent(i) and len(c.countries)<size:
        smallest = c
        size = len(c.countries)
    return smallest
  
  def getSmallestPositiveEmptyCont(self):
    '''! Return the smallest continent that is completely empty and has positive bonus
    '''
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if self.world.isEmptyContinent(i) and len(c.countries)<size and c.bonus>0:
        smallest = c
        size = len(c.countries)
    return smallest
    
  def getSmallestOpenCont(self):
    '''! Return the smallest continent that has at least one empty country
    '''
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if self.world.isOpenContinent(i) and len(c.countries)<size:
        smallest = c
        size = len(c.countries)
    return smallest
    
  def getSmallestPositiveOpenCont(self):
    '''! Return the smallest continent that has positive bonus and at least one empty country
    '''
    smallest = None
    size = sys.maxsize
    for i, c in self.world.continents.items():
      if self.world.isOpenContinent(i) and len(c.countries)<size and c.bonus>0:
        smallest = c
        size = len(c.countries)
    return smallest
    
  def closestCountryWithOwner(self, country:int, owner:int):
    '''! Needs testing
    '''
    G = self.world.map_graph
    paths = nx.single_source_shortest_path(G, country, cutoff=G.num_nodes)
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
    Could use networkx for that
    '''
    pass
    
  def easyCostFromCountryToContinent(self, country:int, cont:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    Could use networkx for that
    '''
    pass
   
  def easyCostBetweenCountries(self, source:int, target:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    Could use networkx for that
    '''
    pass
    
  def friendlyPathBetweenCountries(self, source:int, target:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    Could use networkx for that
    '''
    pass
    
  def cheapestRouteFromOwnerToCont(self, owner:int, cont:int):
    '''
    Have to build an auxiliary graph weighted with the armies in target, to find shortest paths
    Could use networkx for that
    '''
    pass
  
#%% Methods I have found useful on the run


  def getAttackListTarget(self, target:int):
    '''! Countries that can attack the target
    '''
    return list(self.world.predecessors(target))
    
  def getAttackListSource(self, source:int):
    '''! Countries that can be attacked from source
    '''
    return list(self.world.successors(source))
    
  def getCountriesPlayer(self, player:int): 
    '''! Return a list of countries currently owned by player
    '''
    return [c for c in self.countries() if c.owner==player]
    
  def __deepcopy__(self, memo):
    act_code = self.activePlayer.code
    new_world = copy.deepcopy(self.world, memo)
    new_players = [p.__class__() for i, p in self.players.items()]   
    new_board = Board(new_world, new_players)
    
    # Copy everything for the players
    for i, p in new_board.players.items():
      for n in self.players[i].__dict__:
        if n != 'board':
          setattr(p, n, copy.deepcopy(getattr(self.players[i], n), memo))
    
    new_board.setPreferences(self.prefs)
    # Extra copying
    other_attrs = ['deck', 'cardSequence', 'nextCashArmies']
    for a in other_attrs:
      setattr(new_board, a, copy.deepcopy(getattr(self, a)))
      
    # Active player must be the same, and playerCycle at the same player
    while new_board.activePlayer.code != act_code:
      new_board.activePlayer = next(new_board.playerCycle)   
    new_board.gamePhase = self.gamePhase
    return new_board
    
    
  def simulate(self, newAgent, playerCode, changeAllAgents = True, maxRounds = 60,
               safety=10e5, sim_console_debug = False):
    '''! Use to facilitate the playouts for search algorithms. Should be called from a copy of the actual board, because it would change the game.
    The player that called the simulation should give a new agent representing a policy it would follow, so that in the new copy the player will be changed with this new agent, and the game will be played until the end or for a maximum number of rounds.
    '''
    oldActivePlayerCode = self.activePlayer.code

    # Change first the player that executed the simulation
    # with its default policy
    newPlayer = self.copyPlayer(playerCode, newAgent)
    self.players[playerCode] = newPlayer  
    self.players[playerCode].setPrefs(playerCode)
    self.players[playerCode].human = False
    
    # Change all players to newAgent to do the rollout
    for i, p in self.players.items():
      if changeAllAgents or p.human:
        p.console_debug = sim_console_debug
        newPlayer = self.copyPlayer(i, newAgent) 
        self.players[i] = newPlayer  
        self.players[i].setPrefs(i)
        self.players[i].human = False
    
      
    self.playerCycle = itertools.cycle(list(self.players.values()))
    self.activePlayer = next(self.playerCycle)
    while self.activePlayer.code != oldActivePlayerCode:
      self.activePlayer = next(self.playerCycle)
    self.prepareStart()
    
    # Simulate the game    
    initRounds = self.roundCount
          
    cont = 0
    self.console_debug = sim_console_debug
    while not self.gameOver and self.roundCount-initRounds < maxRounds and cont < safety:      
      self.play()
      cont += 1 # for safety      
      

  def encodePlayer(self, p):
    return f'{p.code}_{len(p.cards)}'
    
  def encode(self):
    countries = '_'.join([c.encode() for c in self.countries()])
    players = '_'.join([self.encodePlayer(p) for _, p in self.players.items() if p.is_alive])
    return countries + '-' + players

  def __hash__(self):
    return hash(self.encode())

  def __repr__(self):
    '''! Gives a String representation of the board. '''
    return self.encode()
   
  
  def report(self):
    print('------- Board report --------')
    print('Board id :', self.board_id)
    print('Game phase :', self.gamePhase)
    print("Round count: ", self.roundCount)
    print("Active player: ", self.activePlayer.code, self.activePlayer.name())
    print('------- Players --------')
    for i, p in self.players.items():
      num_conts = sum([(c.owner==p.code) for i, c in self.world.continents.items()])
      bonus = sum([(c.owner==p.code)*int(c.bonus) for i, c in self.world.continents.items()])
      print(f'\t{i}\t{p.name()}\t{p.__class__}\n\t\t: initialArmies={p.initialArmies}, income={p.income}, armies={self.getPlayerArmies(p.code)}, countries={p.num_countries}')
      print(f'\t\t: cards={len(p.cards)}, continents={num_conts}, bonus = {bonus}')
    
      
  def getAllPlayersArmies(self):
    return [self.getPlayerArmies(p.code) for i,p in self.players.items()]
  
  def getAllPlayersNumCountries(self):
    return [p.num_countries for i,p in self.players.items()]
  
  def getCountriesPlayerThatCanAttack(self, player:int):
    cc = self.getCountriesPlayer(player)
    res = []
    for c in cc:
      if len(self.world.getCountriesToAttack(c.code))>0 and c.armies>1:
        res.append(c)
    return res
  
  def getCountriesPlayerWithEnemyNeighbors(self, player:int, kind=1):
    cc = self.getCountriesPlayer(player)
    res = []
    for c in cc:
      if len(self.world.getCountriesToAttack(c.code))>0:
        res.append(c)
    return res
  
  def replacePlayer(self, player:int, newAgent):
    oldActivePlayerCode = self.activePlayer.code

    newPlayer = self.copyPlayer(player, newAgent) 
    self.players[player] = newPlayer      
    self.players[player].setPrefs(player)
        
    self.playerCycle = itertools.cycle(list(self.players.values()))
    self.activePlayer = next(self.playerCycle)
    while self.activePlayer.code != oldActivePlayerCode:
      self.activePlayer = next(self.playerCycle)
    self.prepareStart()
    
    
  def copyPlayer(self, player:int, newAgent=None):
    oldPlayer = self.players[player]
    if newAgent is None or newAgent.human:
      newPlayer = oldPlayer.__class__()
    else:
      newPlayer = newAgent.__class__()
      
    for n in oldPlayer.__dict__:
      if n != 'name_string':
        setattr(newPlayer, n, copy.deepcopy(getattr(oldPlayer,n)))
    newPlayer.console_debug = False
    return newPlayer
  
  def countriesPandas(self):
    df = pd.DataFrame(data = {'id':[c.id for c in board.countries()],
                          'name':[c.name for c in board.countries()],
                          'continent':[c.continent for c in board.countries()],
                          'owner':[c.owner for c in board.countries()],
                          'armies':[c.armies for c in board.countries()]})
    return df

#%% TESTING
if __name__ == '__main__':

  console_debug = True
  
  # Load map
  path = '../support/maps/classic_world_map.json'
  path = '../support/maps/test_map.json'
    
  world = World(path)
  

  # Set players
  aggressiveness = 0.7
  pH, pR1, pR2 = agent.Human('PocketNavy'), agent.RandomAgent('Red',aggressiveness), agent.RandomAgent('Green',aggressiveness)
  pMC = agent.FlatMC('MC', agent.RandomAgent, budget=300)
  pMC2 = agent.FlatMC('MC', agent.RandomAgent, budget=300)
  pP = agent.PeacefulAgent()
  pA = agent.RandomAggressiveAgent('RandomAgressive')
 
  players = [pR1, pMC]
  # Set board
  prefs = {'initialPhase': False, 'useCards':True,
           'transferCards':True, 'immediateCash': True,
           'continentIncrease': 0.05, 'pickInitialCountries':True,
           'armiesPerTurnInitial':4,'console_debug':False}  
  board = Board(world, players)
  board.setPreferences(prefs)
  
  #%% Test play
  if False:
    import tqdm
    
    N = 60
    start = time.process_time()
    iter_times = []
    armies = []
    countries = []
    board.console_debug = True
    
    pMC.budget = 500
    pMC.inner_placeArmies_budget = 100
    pMC.console_debug = True
    
    for i in range(N):
      in_start = time.process_time()
      board.play()
      board.report()
      print('-'*30)
      
      user_msg = input("Press q to quit, c to continue: ")
      while not user_msg in ['q', 'c']:
        user_msg = input("Press q to quit, c to continue: ")
      if user_msg == 'q': break
      print('-'*30)
      iter_times.append(time.process_time()-in_start)
      armies.append(board.getAllPlayersArmies())
      countries.append(board.getAllPlayersNumCountries())
      if board.gameOver: break
    total = time.process_time()-start
    
    armies = np.array(armies)
    countries = np.array(countries)
    
    print(f"Total time: {total} sec \t Avg time per play: {total/N}")
    board.report()
    
    import matplotlib.pyplot as plt
    for i, p in enumerate(players):
      plt.plot(armies[:,i], label=p.name())
    plt.legend()
    plt.title("Armies")
    plt.show()
    
    for i, p in enumerate(players):
      plt.plot(countries[:,i], label=p.name())
    plt.legend()
    plt.title("Countries")
    plt.show()
  
  #%% Test deepcopy 
  if False:
    board_copy = copy.deepcopy(board)
    for i in range(10):
      board.play()
      board_copy.play()
    board.report()
    board_copy.report()
    for i in range(5):
      board_copy.play()
    print("\n\nSee one country")
    print(board.getCountryById('BRA'))
    print(board_copy.getCountryById('BRA'))
    
    print("\n\nWorld reference")
    print(board.world)
    print(board_copy.world)
    
    print("\n\nPlayers")
    for i, p in board.players.items():
      print(i, p)
      print('Cards:', p.cards)
    for i, p in board_copy.players.items():
      print(i, p)
      print('Cards:', p.cards)
      
      
    print("\n\nDecks")
    print("\nboard")
    for c in board.deck.deck:
      print(c)
    print("\nboard_copy")
    for c in board_copy.deck.deck:
      print(c)
      
    print("Draw one card", board.deck.draw())
    print("\n\nDecks")
    print("\nboard")
    for c in board.deck.deck:
      print(c)
    print("\nboard_copy")
    for c in board_copy.deck.deck:
      print(c)
  
  #%% Test simulate
  if False:
    pMC.console_debug = False
    players = [pMC, pR1]
    board = Board(world, players)
    board.setPreferences(prefs)
    for i in range(2):
      board.play()
    board.report()

    for j in range(5):
      sim_board = copy.deepcopy(board)

      # newAgent (instance, not class), playerCode
      
      sim_board.simulate(agent.RandomAgent(),pMC.code, changeAllAgents=False,
                         maxRounds=20, safety = 10e5, sim_console_debug=False)
      print(f"\nBoard after simulation {j}\n")
      sim_board.report()
      print("\nScore for 0")
      pMC2.setPrefs(0)
      print(pMC2.score(sim_board))
      print("\nScore for 1")
      pMC2.setPrefs(1)
      print(pMC2.score(sim_board))
      
    
  
    
  #%% Test fromDicts, toDicts
  if False:
    for _ in range(3):
      board.play()
    continents, countries, inLinks, players, misc = board.toDicts()
    board_copy = Board.fromDicts(continents, countries, inLinks,\
                                 players, misc)
      
    print(board.countries())
    print(board_copy.countries())
    print()
    print(board.activePlayer.code)
    print(board_copy.activePlayer.code)

  test_dict = {}
  test_dict[board] = {}
  c = board.countries()[0]
  move = agent.Move(c,c,1,'startTurn')
  test_dict[board][move] = 'testing'
  move = agent.Move(c,c,1,'attack')
  test_dict[board][move] = 'testing 2'
  
  test_dict[board] = 5
  board.gamePhase = 'other'
  test_dict[board] += 5
  print(test_dict)
  
    
