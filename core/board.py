import sys
import itertools
import numpy as np
import copy
import pandas as pd
import random

from deck import Deck, ListThenArithmeticCardSequence
from world import World, Country
from move import Move

from misc import iter_all_strings

class Agent(object):
  '''!Base class for an agent
  
  All players should extend from this class
  '''  
  def __init__(self, name='agent'):
    '''! Constructor. By default sets the *console_debug* and *human* 
    attributes to **False**
    
    :param name: The name of the agent
    :type name: str
    '''
    self.name = name
    self.human = False
    self.console_debug = False

  def __deepcopy__(self, memo):
    newPlayer = self.__class__()
    for n in self.__dict__:
      setattr(newPlayer, n, getattr(self, n))
    return newPlayer

  def copyToAgent(self, newPlayer):
    for n in self.__dict__:
      setattr(newPlayer, n, getattr(self, n))
  
  def setPrefs(self, code:int):
    '''! Puts the agent into the game by assigning it a code and giving it
    the reference to the game board
    
    :param code: The internal code of the agent in the game.
    :type code: int
    :param board: Reference to the board with the game information
    :type :py:module:`pyRisk`.:py:class:`Board`
    '''
    self.code = code   
    
  
  def pickCountry(self, board) -> int:
    '''! Choose an empty country at the beginning of the game
    '''
    pass
    
  def placeInitialArmies(self, board, numberOfArmies:int):
    '''! Place armies in owned countries. No attack phase follows.
    Use board.placeArmies(numberArmies:int, country:Country)
    
    :param numberOfArmies: The number of armies the agent must place on the
    board
    :type numberOfArmies: int
    '''
    pass
  
  
  def cardPhase(self, board, cards):
    '''! Call to exchange cards.
    May return None, meaning no cash
    Use :py:module:`pyRisk`.:py:module:`Deck` methods to check for sets and
    get best sets
    
    :param cards: List with the player's cards
    :type cards: list[:py:module:`pyRisk`.:py:class:`Card`]
    :returns List of three cards to cash
    :rtype list[:py:module:`pyRisk`.:py:class:`Card`]
    '''
    pass
  
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Given a number of armies, place them on the board
    Use board.placeArmies(numberArmies:int, country:Country) to place the
    armies
    
    :param numberOfArmies: The number of armies the agent must place on the
    board
    :type numberOfArmies: int
    '''
    pass
  
  
  def attackPhase(self, board):
    '''! Call to attack. 
    Can attack till dead or make single rolls.
    Use res = board.attack(source_code:int, target_code:int, tillDead:bool)
    to do the attack
    Every call to board.attack yields a result as follows:
      - 7: Attacker conquered the target country. This will call to
      moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int)
      to determine the number of armies to move to the newly conquered
      territory
      - 13: Defender won. You are left with only one army and can therefore
      not continue the attack
      - 0: Neither the attacker nor the defender won. If you want to deduce
      the armies you lost, you can demand the number of armies in your
      country.
      - -1: There was an error
      
    You can attack multiple times during the attack phase.
    '''
    pass
    
  def moveArmiesIn(self, board, countryCodeAttacker:int, countryCodeDefender:int) -> int:
    '''! This method is called when an attack led to a conquer. 
    You can choose the number of armies to send to the conquered territory. 
    The returned value should be between the number of rolled dice and the
    number of armies in the attacking country minus 1.
    The number of dice rolled is always 3 except when you have 3 or less
    armies, in which case it will be the number of attacking armies minus 1.
    Notice that by default, this method returns all the moveable armies.
    
    
    :param countryCodeAttacker: Internal code of the attacking country
    :type countryCodeAttacker: int
    :param countryCodeDefender: Internal code of the defending country
    :type countryCodeDefender: int
    :returns Number of armies to move from attacking country to newly
    conquered territory.
    :rtype int
    '''
    # Default is to move all but one army
    return board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self, board):
    '''! Call to fortify.
    Just before this call, board will update the moveable armies in each
    territory to the number of armies.
    You can fortify multiple times, always between your countries that are
    linked and have moveable armies.
    '''
    pass
  def youWon(self):
    return "This agent won the game"

class RandomAgent(Agent):
  """ Base agent
  """

  def __init__(self, name='random'):
    '''! Constructor of random agent.
    
    :param name: Name of the agent.
    :type name: str
    :param aggressiveness: Level of aggressiveness. Determines the probability
    of attacking until dead. 1 means always attacking until dead when
    attacking.
    :type aggressiveness: float
    '''
    self.name = name
    self.human = False
    self.console_debug = False    
 
  def choose_at_random(self, board):
    options = board.legalMoves()
    if not options:
      print("\n\nNo legal moves found")
      board.report()
      print(board.countriesPandas())
      print(f"Player: {board.activePlayer.name} ({board.activePlayer.code})")
      print(f"Player is alive: {board.activePlayer.is_alive} (num countries {board.activePlayer.num_countries})")
    return np.random.choice(options)
    
  
  def pickCountry(self, board):
    '''! Pick at random one of the empty countries
    '''    
    return self.choose_at_random(board)
    
  
  def placeInitialArmies(self, board, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''  
    return self.choose_at_random(board)
    
    
  def cardPhase(self, board, cards):
    '''! Only cash when forced, then cash best possible set
    '''
    if not Deck.containsSet(cards): 
      return None
    elif 0.5 < np.random.uniform():
      c = Deck.yieldBestCashableSet(cards, self.code, board.world.countries)
      if not c is None:      
        return c
    else: 
      return None
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Place armies at random one by one, but on the countries with enemy
    borders
    '''
    return self.choose_at_random(board)
  
  def attackPhase(self, board):
    '''! Attack a random number of times, from random countries, to random
    targets.
    The till_dead parameter is also set at random using an aggressiveness
    parameter
    '''  
    return self.choose_at_random(board)
    
  def moveArmiesIn(self, board, countryCodeAttacker:int, countryCodeDefender:int) -> int:
    '''! Move a random number or armies. If the choice is wrong, the board will fix it
    Wrong means > than armies-1 or < aDice
    '''
    # TODO: Communicate the number of dice used
    # This also means letting the player choose the number of dice
    return np.random.choice(board.world.countries[countryCodeAttacker].armies)
  
  def fortifyPhase(self, board):
    '''! For now, no fortification is made
    '''
    return self.choose_at_random(board)
        


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
    self.initialArmies = initialArmies
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
    num_wildcards = len(self.world.countries)//20*0
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
      p = RandomAgent(name = f"Player {i}")
      new_players[i] = p
      
    board = Board(world, list(new_players.values()))
    prefs = misc.get('prefs')
    if not prefs is None:
      board.setPreferences(prefs)
      
    for i, attrs in players.items():
      for n in range(int(attrs['cards'])):
        new_players[i].cards.append(board.deck.draw())
        
    board.gamePhase = misc['gamePhase']
    while board.activePlayer.code != int(misc['activePlayer']):
      board.activePlayer = next(board.playerCycle)
      
    while board.nextCashArmies != int(misc['nextCashArmies']):
      board.nextCashArmies = board.cardSequence.nextCashArmies()
      if board.nextCashArmies > int(misc['nextCashArmies']):
        raise Exception("Board:fromDicts: Error with nextCashArmies. The value was not a value obtained from the default card sequence")
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
    
    for c in self.countries():
      c.attrToDict()
    
    for _, c in self.world.continents.items():
      c.attrToDict()
      
    countries = {c['code']: c for c in self.countries()}
    continents = {i: c for i,c in self.world.continents.items()}
    players = dict()
    for i, p in self.players.items():      
      players[p.code] = {'code':p.code, 'name':p.name,
                       'income': self.getPlayerIncome(p.code),
                       'cards':len(p.cards),
                       'alive':p.is_alive,
                       'countries':p.num_countries}      
    inLinks = {n.code: [c.code for c in self.world.predecessors(n.code)] for n in self.countries()}
    misc = {'gamePhase': self.gamePhase, 'activePlayer':self.activePlayer.code,
            'nextCashArmies':self.nextCashArmies}
    return continents, countries, inLinks, players, misc
    
  def toCanonical(self, rootPlayer):
    ''' The idea is that rootPlayer becomes player 0,the rest are moved accordingly
        Returns a copy of the current board with the canonical representation
    '''
    rootPlayer = int(rootPlayer)
    new = copy.deepcopy(self)
    numPlayers = len(self.players)
    if rootPlayer == 0: return new, {i: i for i in range(numPlayers)}
    
    map_players = {(rootPlayer + i)%numPlayers: i for i in range(numPlayers)}
    map_to_orig = {i: (rootPlayer + i)%numPlayers for i in range(numPlayers)}
    map_players[-1] = -1
    # Update countries
    for c in new.countries():
      c.owner = map_players[c.owner]
    
    # Update players dictionary  
    copy_players = copy.deepcopy(new.players)
    for i in range(numPlayers):
      new.players[map_players[i]] = copy_players[i]
      new.players[map_players[i]].code = map_players[i]
      new.players[map_players[i]].setPrefs(map_players[i])
    
    # Update activePlayer and playerCycle
    newActive = map_players[self.activePlayer.code]
    new.playerCycle = itertools.cycle(list(new.players.values()))
    while new.activePlayer.code != newActive:
      new.activePlayer = next(new.playerCycle)
    
    new.firstPlayerCode = map_players[self.firstPlayerCode]
    new.lastPlayerCode = map_players[self.lastPlayerCode]
    
    
    # Update continents
    new.updateContinents()
    return new, map_to_orig
  
  def setPreferences(self, prefs):
    self.prefs = prefs
    available_prefs = ['initialArmies', 'initialPhase', 'useCards', 'transferCards', 'immediateCash', 'continentIncrease', 'pickInitialCountries', 'armiesPerTurnInitial', 'console_debug']
    # In the future, card sequences, num_wildcards, armiesInitial
    for a in available_prefs:
      r = prefs.get(a)
      if r is None:
        print(f"Board preferences: value for '{a}' not given. Leaving default value {getattr(self, a)}")
      else:
        setattr(self, a, r)
    if 'initialArmies' in prefs:
      init = prefs['initialArmies']
      for _, p in self.players.items():       
        p.initialArmies = int(init)        
        
#%% Board: Play related functions
  
  # See class Move to see how a move is modeled
  def legalMoves(self):
    '''! Given a board, creates a list of all the legal moves
    '''
    p = self.activePlayer
    armies = p.income
    
    # PICK COUNTRY
    if self.gamePhase == 'initialPick':
      return [Move(c,c,0, 'initialPick') for c in self.countriesLeft()]
      
    # PLACE ARMIES
    elif self.gamePhase in ['initialFortify', 'startTurn']:
      if armies == 0:
        print("Board:legalMoves: No armies to place??")
        self.report()
        self.showPlayers()
        return [Move(None, None, None, self.gamePhase)]
      
      return [Move(c,c,a, self.gamePhase) for c,a in itertools.product(self.getCountriesPlayer(p.code), range(armies,-1,-1))]
      
      # Simplify: Place everything in one country
      # return [Move(c,c,a, self.gamePhase) for c,a in itertools.product(self.getCountriesPlayer(p.code), [0])]
    
    # ATTACK
    elif self.gamePhase == 'attack':
      moves = []
      for source in self.getCountriesPlayerThatCanAttack(p.code):
        for target in self.world.getCountriesToAttack(source.code):
          # Attack once
          moves.append(Move(source, target, 0, 'attack'))
          # Attack till dead
          moves.append(Move(source, target, 1, 'attack'))
      # Always possible to just pass
      moves.append(Move(None, None, None, 'attack'))
      return moves
    
    # FORTIFY
    elif self.gamePhase == 'fortify':    
      # For the moment, only considering to fortify 5 or all
      moves = []
      for source in self.getCountriesPlayer(p.code):
        for target in self.world.getCountriesToFortify(source.code):          
          if source.moveableArmies > 0 and source.armies > 1:
            # Fortify all or 1
            moves.append(Move(source, target, 0,'fortify'))            
            
            # Simplify
            moves.append(Move(source, target, 1,'fortify'))
          
          if source.moveableArmies > 5 and source.armies > 1:
            # Simplify
            moves.append(Move(source, target, 5,'fortify'))
            
            
      # Always possible to just pass
      moves.append(Move(None, None, None, 'fortify'))
      return moves
      

  def playMove(self, move):
    '''! Simplifies the playing of a move by considering the different game phases
    '''
    # INITIAL PICK
    if self.gamePhase == 'initialPick':
      
      if self.console_debug: print(f"Board {self.board_id} PlayMove:initialPick: {move.source.id}")
    
      # Make action
      res = self.pickCountry(move.source.code)
      
      # Check for errors
      if res == -1: return -1
      
      # Check if game phase changes
      if len(self.countriesLeft())==0:      
          self.gamePhase = 'initialFortify'
          
      # In any case, after picking a country, the player turn ends
      self.endTurn()
      return res
    
    # INITIAL FORTIFY
    elif self.gamePhase == 'initialFortify':
      
      if self.console_debug: print(f"Board {self.board_id} PlayMove:initialFortify: {move.source.id}, {move.details}")
      
      res = self.placeArmies(int(move.details), move.source.code)
      
      # Check for errors
      if res == -1: return -1
      
      # Check if initial fortify is done
      if(sum([p.initialArmies + p.income for i, p in self.players.items()])==0):
        self.gamePhase = 'startTurn'
      
      # If the phase is not over, must check if it is time to change player
      # If the phase is over, ready up for startTurn phase
      if self.activePlayer.income == 0:
        self.endTurn()

    # START TURN - PLACE ARMIES
    elif self.gamePhase == 'startTurn':   
      
      if self.console_debug: print(f"Board {self.board_id} PlayMove:startTurn: {move.source.id}, {move.details}")        
      
      res = self.placeArmies(int(move.details), move.source.code)
      
      # Check for errors
      if res == -1: return -1
      
      # Check if placing armies is done
      if self.activePlayer.income == 0:
        self.gamePhase = 'attack'      
      return res
      
    # ATTACK
    elif self.gamePhase == 'attack':
      if self.console_debug: 
        if not move.source is None: 
          print(f"Board {self.board_id} PlayMove:Attack: {move.source.id} ({move.source.armies}), {move.target.id} ({move.target.armies}), tillDead: {bool(move.details)}")
        else:
          print("PlayMove:Attack: PASS")
      # Pass move, do not attack
      if move.source is None:      
          self.updatemoveable()
          self.gamePhase = 'fortify'
          return 0
          
      # Try to perform the attack
      try:
        
        return self.attack(move.source.code, move.target.code, bool(move.details))
      except Exception as e:
        raise e
        
    # FORTIFY
    elif self.gamePhase == 'fortify': 
      if self.console_debug: 
        if not move.source is None:
          print(f"Board {self.board_id} PlayMove:Fortify: {move.source.id}, {move.target.id}, {move.details}")
        else:
          print("PlayMove:Fortify: PASS")
    
      pass_move = False
      if move.source is None or move.target is None:
        pass_move = True
      elif move.source.moveableArmies == 0 or move.source.armies ==1:
        pass_move = True
      
      if pass_move:
        self.gamePhase = 'startTurn'
        self.endTurn()
        if self.console_debug: print(f"Board {self.board_id} PlayMove:Fortify: Ending turn. Active player {self.activePlayer.code}, First player {self.firstPlayerCode}")
        if self.activePlayer.code == self.firstPlayerCode:
          # New round has begun
          if self.console_debug: print(f"Board {self.board_id} PlayMove:Fortify: New round")
          self.setupNewRound()
          self.prepareStart()
          
        return 0
      
      # Perform a fortify
      res = self.fortifyArmies(int(move.details), move.source.code, move.target.code)      
      return res
      
  def isTerminal(self, rootPlayer):
    if self.getNumberOfPlayersLeft()==1:
      if self.players[rootPlayer].is_alive: return 1
      return -1
    # Not over
    return 0
  
  
  def play(self):
    '''! Function that represents a move. 
    The board asks the active player for a move, receives the move and plays it.
    It then updates the state of the game and returns
    Notice that one turn will usually take several moves
    '''
    
    if self.gameOver:
      if self.console_debug: print("Board:play: Game over")
      return
    
    # If there is no initial phase
    if not self.pickInitialCountries or not self.initialPhase:
      if self.gamePhase == 'initialPick':
        if self.console_debug: print("Board:play: Random initial Pick")
        # Select countries at random
        countries = self.countries()
        random.shuffle(countries)
        moves = [Move(c,c,0, 'initialPick') for c in countries]
        for m in moves: self.playMove(m)
        return
        
      if self.gamePhase == 'initialFortify' and not self.initialPhase:
        if self.console_debug: print("Board:play: Random initial fortify")
        while self.gamePhase == 'initialFortify':
          c = np.random.choice(self.getCountriesPlayer(self.activePlayer.code))
          self.playMove(Move(c,c,1, 'initialFortify'))
          return
          
    
    
    p = self.activePlayer
    if self.console_debug: print(f"\n***Board:play: Starting play for ({p.code}), {p.name}")
    
    if not p.is_alive:
      if self.console_debug: print("Board:play: Returning, player is not alive")
      self.endTurn()

    else:   
      
      # Initial phase
      if self.gamePhase == 'initialPick':
        if self.console_debug: print(f"Board:play: initialPick, player {p.name} to pick... ")
        move = p.pickCountry(self)
        if self.console_debug: print(f"Board:play: initialPick: Picked {move}")
        return self.playMove(move)
      
      # Initial Fortify
      if self.gamePhase == 'initialFortify':
        if self.console_debug: print(f"Board:play: initialFortify, player {p.name} to fortify... ")
        # Check if player has initial armies left. If not, is the phase over or not?
        if p.income + p.initialArmies == 0:
          if self.console_debug: print(f"Board:play: initialFortify, player {p.name} has no armies left to place, ending turn")
          self.endTurn()
          return 0
                       
        move = p.placeInitialArmies(self, p.income)        
        if self.console_debug: print(f"Board:play: initialFortify: Fortify {move}")
        return self.playMove(move)
        
      # Start turn: Give armies and place them
      if self.gamePhase == 'startTurn':  
        if self.console_debug: print(f"Board:play: {self.gamePhase}")
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
                raise Exception(f"More than 5 cards but not able to find a cashable set. Maybe the deck has too many wildcards? -- {p.cards}")
          armies += cashed          
        self.activePlayer.income += cashed
        move = p.placeArmies(self, armies)
        return self.playMove(move)
       
        
      # Attack
      if self.gamePhase == 'attack':
        if self.console_debug: print(f"Board:play: {self.gamePhase}")        
        move = p.attackPhase(self)     
        return self.playMove(move)
              
      # Fortify
      if self.gamePhase == 'fortify':
        if self.console_debug: print(f"Board:play: {self.gamePhase}")        
        move = p.fortifyPhase(self)
        return self.playMove(move)

#%% Methods for the game phases  
  def pickCountry(self, country_code:int) -> int:
    c = self.world.countries.get(country_code)
    if c is None: return -1
    if c.owner != -1: return -1
    
    p = self.activePlayer    
    c.owner = p.code
    c.armies += 1    
    p.initialArmies -= 1
    p.num_countries += 1
    if self.console_debug: print(f"pickCountry: Player {p.code} picked {c.id}")
    return 0
    
  def cashCards(self, card, card2, card3):
    '''! Cashes in the given card set. Each parameter must be a reference to a different Card instance sent via cardsPhase(). 
    It returns the number of cashed armies'''
    if Deck.isSet(card, card2, card3):
      res = self.nextCashArmies
      self.nextCashArmies = self.cardSequence.nextCashArmies()
      if self.console_debug:
        print(f'Board:cashCards:Obtained {res} armies for player {self.activePlayer.code}')
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
          print(f'Board:cashCards: Card {c} does not belong to activePlayer {self.activePlayer.code}, {self.activePlayer.name}: {self.activePlayer.cards}')
          raise e      
      return res
    else:
      return 0 
  
  def force_card_cash(self):
    p = self.activePlayer
    card_set = Deck.yieldCashableSet(p.cards)
    if card_set is None:      
      return 0
    else:
      return self.cashCards(*card_set)
    
  def placeArmies(self, numberOfArmies:int, country) -> int:
    '''! Places numberOfArmies armies in the given country. '''
    code = country.code if isinstance(country, Country)  else country
    if self.world.countries[code].owner != self.activePlayer.code: return -1    
    if self.activePlayer.income < numberOfArmies: return -1
    
    # Good to go. Check if numberOfArmies == 0 which means to put all available armies
    if numberOfArmies == 0: numberOfArmies = self.activePlayer.income
    if self.console_debug:
      p = self.activePlayer
      print(f'placeArmies: Player {p.code}, {numberOfArmies} armies in {self.world.countries[code].id}. {p.income-numberOfArmies} armies left to place')
    self.world.countries[code].addArmies(numberOfArmies)
    self.activePlayer.income -= numberOfArmies
    return 0
    
  def roll(self, attack:int, defense:int):
    '''! Simulate the dice rolling. Inputs determine the number of dice to use for each side.
    They must be at most 3 for the attacker and 2 for the defender.
    Returns the number of lost armies for each side
    '''
    if attack > 3 or attack <= 0: return None
    if defense > 2 or defense <= 0: return None
    aDice = sorted(np.random.randint(1,7,size=attack), reverse=True)
    dDice = sorted(np.random.randint(1,7,size=defense), reverse=True)
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
      if self.console_debug: print(f'attack:Dice roll - attacker loss {aLoss},  defender loss {dLoss}')
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
          # Add cards to use in next turn
          attacker.cards.extend(defender.cards)
          
          if self.getNumberOfPlayersLeft() == 1: 
            self.gameOver = True
            return 99
          
          # To simplify, cards won by eliminating an opponent can only be used on the next turn, in the startTurn phase
                 
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
    if cO.moveableArmies == 0 or cO.moveableArmies < numberOfArmies: return -1
 
    # Even if moveableArmies == armies, we have to always leave one army at least    
    if numberOfArmies == 0: numberOfArmies = cO.moveableArmies
    aux = numberOfArmies-1 if numberOfArmies == cO.armies else numberOfArmies
    
    if self.console_debug:
      print(f'Board:Fortify: From {cO.id} to {cD.id}, armies = {aux}')
    cO.armies -= aux
    cO.moveableArmies -= aux
    cD.armies += aux
    return 1
    
   
#%% Functions to update information, end turns and start new rounds
  
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
      p.income = min(p.initialArmies, self.armiesPerTurnInitial)
      p.initialArmies -= p.income
  
  def setupNewRound(self):
    '''! Performs the necessary actions at the end of a round (all players played)
    '''
    self.roundCount += 1
    if self.roundCount != 1:
      # Update continent bonus if it is not the first round
      for i, cont in self.world.continents.items():
        cont.bonus = cont.bonus*(1+self.continentIncrease)
    
    s = 0
    for i, p in self.players.items():
      if p.is_alive: s += 1
    
    if s == 1: self.gameOver = True
    if self.console_debug: print(f"Board:setupNewRound: Round {self.roundCount}. Found {s} players alive. gameOver = {self.gameOver}")
    return
      
  def prepareStart(self):
    '''! At the very beginning of a player's turn, this function should be called to update continent ownership and active player income.
    '''
    if self.gamePhase == 'startTurn' or self.gamePhase == 'initialFortify':      
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
    if self.console_debug: print(f"Board:endTurn: Ending turn: next player is ({self.activePlayer.code}) {self.activePlayer.name}")
    
    #if self.console_debug: print(f"Player info: income = {p.income}")
    self.prepareStart()
    #if self.console_debug: print(f"Player info: income = {p.income}, game phase = {self.gamePhase}")
  
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
     
  
  
#%% Simulation

  def readyForSimulation(self):
    activePlayerCode = self.activePlayer.code
    # Change all players to random    
    for i, p in self.players.items():
      self.players[i] = self.copyPlayer(i, newAgent=RandomAgent())
    # Fix player cycle
    self.playerCycle = itertools.cycle(list(self.players.values()))
    while self.activePlayer.code != activePlayerCode:
      self.activePlayer = next(self.playerCycle)
    
  
  
  def simulate(self, maxRounds = 60, safety=10e5, sim_console_debug = False):
    '''! Use to facilitate the playouts for search algorithms. 
    Should be called from a copy of the actual board, because it the board will be modified.    
    '''
    
    # Simulate the game    
    initRounds = self.roundCount
    
    cont = 0
    self.console_debug = sim_console_debug
    
    self.readyForSimulation()
    
    #print("Before starting sim:")
    #self.showPlayers()
    #self.report()
      
    while not self.gameOver and self.roundCount-initRounds < maxRounds and cont < safety:      
      self.play()
      cont += 1 # for safety 
  
  
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
      return p.name
  
  def getAgentName(self, player:int) -> str:
    '''! Returns whatever the name method of the of the given agent returns.  '''
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
  
  def getPlayerContinents(self, player:int)-> int:
    '''! Return the number of countries a player owns
    '''    
    self.updateContinents()    
    s = 0
    for i, c in self.world.continents.items():
      if c.owner == player:
        s += 1
    return s
  
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
    p = self.players.get(player)
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
  
#%% Methods I have found useful on the run


  def getAttackListTarget(self, target:int):
    '''! Countries that can attack the target
    '''
    t = self.world.countries[target]
    return [s for s in list(self.world.predecessors(target)) if s.owner != t.owner and s.armies > 1]
    
  def getAttackListSource(self, source:int):
    '''! Countries that can be attacked from source
    '''
    s = self.world.countries[source]
    if s.armies < 2: return []
    return [t for t in list(self.world.successors(source)) if s.owner != t.owner]
    
  def getFortifyListTarget(self, target:int):
    '''! Countries that can attack the target
    '''
    t = self.world.countries[target]
    return [s for s in list(self.world.predecessors(target)) if s.owner == t.owner and s.moveableArmies > 0 and s.armies > 1]
    
  def getFortifyListSource(self, source:int):
    '''! Countries that can be attacked from source
    '''
    s = self.world.countries[source]
    if s.moveableArmies == 0 or s.armies == 1: return []
    return [t for t in list(self.world.successors(source)) if s.owner == t.owner]
    
  def getCountriesPlayer(self, player:int):
    '''! Return a list of countries currently owned by player
    '''
    return [c for c in self.countries() if c.owner==player]
    
    
       
      



  #%% MISC functions
  
  def __deepcopy__(self, memo):
    act_code = self.activePlayer.code
    new_world = copy.deepcopy(self.world, memo)
    new_players = [p.__class__() for i, p in self.players.items()]   
    new_board = Board(new_world, new_players)
    
    # Copy everything for the players
    for i, p in new_board.players.items():
      for n in self.players[i].__dict__:
        if n != 'board':
          # print(f"Board: deepcopy: attr = {n}") # For debugging
          setattr(p, n, copy.deepcopy(getattr(self.players[i], n)))
    
    new_board.setPreferences(self.prefs)
    # Extra copying
    other_attrs = ['deck', 'cardSequence', 'nextCashArmies', 'console_debug']
    for a in other_attrs:
      setattr(new_board, a, copy.deepcopy(getattr(self, a)))
      
    # Active player must be the same, and playerCycle at the same player
    while new_board.activePlayer.code != act_code:
      new_board.activePlayer = next(new_board.playerCycle)   
    new_board.gamePhase = self.gamePhase
    return new_board
    
  
  
  def encodePlayer(self, p):
    return f'{p.code}_{len(p.cards)}'
    
  def encode(self):
    countries = '_'.join([c.encode() for c in self.countries()])
    players = '_'.join([self.encodePlayer(p) for _, p in self.players.items() if p.is_alive])
    return countries + '-' + players + '-'+ self.gamePhase + '-' + str(self.activePlayer.code)

  def __hash__(self):
    return hash(self.encode())

  def __repr__(self):
    '''! Gives a String representation of the board. '''
    return self.encode()
   
  
  def report(self):
    print('\n------- Board report --------')
    print('Board id :', self.board_id)
    print('Game phase :', self.gamePhase)
    print("Round count: ", self.roundCount)
    print("Active player: ", self.activePlayer.code, self.activePlayer.name)
    print('------- Players --------')
    for i, p in self.players.items():
      num_conts = sum([(c.owner==p.code) for i, c in self.world.continents.items()])
      bonus = sum([(c.owner==p.code)*int(c.bonus) for i, c in self.world.continents.items()])
      print(f'\t{i}\t{p.name}\t{p.__class__}\n\t\t: initialArmies={p.initialArmies}, income={p.income}, armies={self.getPlayerArmies(p.code)}, countries={p.num_countries}')
      print(f'\t\t: cards={len(p.cards)}, continents={num_conts}, bonus = {bonus}')
      
  def getAllPlayersArmies(self):
    return [self.getPlayerArmies(p.code) for i,p in self.players.items()]
  
  def getAllPlayersNumCountries(self):
    return [p.num_countries for i,p in self.players.items()]
  
  def getCountriesPlayerThatCanAttack(self, player:int): # GOOD
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
    if newAgent is None:
      newPlayer = oldPlayer.__class__()
    else:
      newPlayer = newAgent.__class__()
    for n in oldPlayer.__dict__:
      setattr(newPlayer, n, copy.deepcopy(getattr(oldPlayer,n)))
    newPlayer.console_debug = False
    return newPlayer
  
  def countriesPandas(self):
    df = pd.DataFrame(data = {'id':[c.id for c in self.countries()],
                          'name':[c.name for c in self.countries()],
                          'continent':[c.continent for c in self.countries()],
                          'owner':[c.owner for c in self.countries()],
                          'armies':[c.armies for c in self.countries()]})
    return df

  def showPlayers(self):
    for i, p in self.players.items():
      print(i, p.code, p)



if __name__ == "__main__":

  # Load a board, try to play

  # Load map
  path = '../support/maps/classic_world_map.json'
  path = '../support/maps/test_map.json'
    
  world = World(path)
  

  # Set players  
  pR1, pR2 = RandomAgent('Red'), RandomAgent('Green')
  
  players = [pR1, pR2]
  # Set board
  prefs = {'initialPhase': True, 'useCards':True,
           'transferCards':True, 'immediateCash': True,
           'continentIncrease': 0.0, 'pickInitialCountries':True,
           'armiesPerTurnInitial':4,'console_debug':False,
           'initialArmies':20}  
           
  board_orig = Board(world, players)
  board_orig.setPreferences(prefs)
  
  board = copy.deepcopy(board_orig)
  
  print("**** See if players are the same on different copies")
  print("board_orig")
  board_orig.showPlayers()
  
  print("\nboard")
  board.showPlayers()
  
  print("\n\n")
  
  # print("**** Test play")
  # board.report()
  # print(board.countriesPandas())
  
  for i in range(1):
    board.play()
    if board.gameOver: break
  
  print("\n\n")  
  board.report()
  print(board.countriesPandas())
  
  print("\n\n")
  
  # print("**** Simulation")
  # board.simulate(maxRounds = 60, safety=10e5, sim_console_debug = True)
  # board.report()
  # print(board.countriesPandas())
  
  # print()
  # print("Board orig")
  
  # board_orig.report()
  # print(board_orig.countriesPandas())
  
  
  print("\n\n\n")
  numTries = 1000
  maxSteps = 2000
  endStep = []
  finished = 0
  for i in range(numTries):
    board = copy.deepcopy(board_orig)
    for j in range(maxSteps):
      board.play()
      if board.gameOver:
        finished += 1
        endStep.append(j)
        break
  
  print(f"Finished games: {finished}")
  print(f"Average end state: {np.mean(endStep)}")
  import matplotlib.pyplot as plt
  plt.hist(endStep)

  

