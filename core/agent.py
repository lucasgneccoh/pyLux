# -*- coding: utf-8 -*-
"""!@package pyRisk

The agent module contains the pyRisk players.
"""

import numpy as np
import pyRisk
import copy
import itertools

class Agent(object):
  '''!Base class for an agent
  
  All players should extend from this class
  '''  
  def __init__(self, name='agent'):
    '''! Constructor. By default sets the *console_debug* and *human* attributes to **False**
    
    :param name: The name of the agent
    :type name: str
    '''
    self.name_string = name
    self.human = False
    self.console_debug = False
  
  def setPrefs(self, code:int, board):
    '''! Puts the agent into the game by assigning it a code and giving it the reference to the game board
    
    :param code: The internal code of the agent in the game.
    :type code: int
    :param board: Reference to the board with the game information
    :type :py:module:`pyRisk`.:py:class:`Board`
    '''
    self.code = code
    self.board = board
    
  
  def pickCountry(self):
    '''! Choose an empty country at the beginning of the game
    '''
    pass
    
  def placeInitialArmies(self,numberOfArmies:int):
    '''! Place armies in owned countries. No attack phase follows.
    Use self.board.placeArmies(numberArmies:int, country:Country)
    
    :param numberOfArmies: The number of armies the agent must place on the board
    :type numberOfArmies: int
    '''
    pass
  
  
  def cardPhase(self, cards):
    '''! Call to exchange cards.
    May return None, meaning no cash
    Use :py:module:`pyRisk`.:py:module:`Deck` methods to check for sets and get best sets
    
    :param cards: List with the player's cards
    :type cards: list[:py:module:`pyRisk`.:py:class:`Card`]
    :returns List of three cards to cash
    :rtype list[:py:module:`pyRisk`.:py:class:`Card`]
    '''
    pass
  
  
  def placeArmies(self, numberOfArmies:int):
    '''! Given a number of armies, place them on the board
    Use self.board.placeArmies(numberArmies:int, country:Country) to place the armies
    
    :param numberOfArmies: The number of armies the agent must place on the board
    :type numberOfArmies: int
    '''
    pass
  
  
  def attackPhase(self):
    '''! Call to attack. 
    Can attack till dead or make single rolls.
    Use res = self.board.attack(source_code:int, target_code:int, tillDead:bool) to do the attack
    Every call to board.attack yields a result as follows:
      - 7: Attacker conquered the target country. This will call to moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int) to determine the number of armies to move to the newly conquered territory
      - 13: Defender won. You are left with only one army and can therefore not continue the attack
      - 0: Neither the attacker nor the defender won. If you want to deduce the armies you lost, you can demand the number of armies in your country.
      - -1: There was an error
      
    You can attack multiple times during the attack phase.
    '''
    pass
    
  def moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int) -> int:
    '''! This method is called when an attack led to a conquer. 
    You can choose the number of armies to send to the conquered territory. 
    The returned value should be between the number of rolled dice and the number of armies in the attacking country minus 1.
    The number of dice rolled is always 3 except when you have 3 or less armies, in which case it will be the number of attacking armies minus 1.
    Notice that by default, this method returns all the movable armies.
    
    
    :param countryCodeAttacker: Internal code of the attacking country
    :type countryCodeAttacker: int
    :param countryCodeDefender: Internal code of the defending country
    :type countryCodeDefender: int
    :returns Number of armies to move from attacking country to newly conquered territory.
    :rtype int
    '''
    # Default is to move all but one army
    return self.board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self):
    '''! Call to fortify.
    Just before this call, board will update the movable armies in each territory to the number of armies.
    You can fortify multiple times, always between your countries that are linked and have movable armies.
    '''
    pass
  
  def name(self):
    '''! Returns the name of the agent
    :returns Name of the agent
    :rtype str
    '''
    return self.name_string
  
  def version(self):
    '''! Returns the version of the agent
    :returns Version of the agent
    :rtype str
    '''
    return '0'
  
  def description(self):
    '''! Returns a short description of the agent
    :returns Short description of the agent
    :rtype str
    '''
    return 'Base description of agent'  
  
  def youWon(self):
    '''! Returns a victory message
    :returns Agent's victory message
    :rtype str
    '''
    return '{} won the game'.format(self.name_string)
  
  def message(self):
    '''! Returns some message. Not being used
    :returns Message
    :rtype str
    '''
    return 'Hello human, this is {}'.format(self.name_string)  


#%% Basic methods and baselines
class RandomAgent(Agent):

  def __init__(self, name='random', aggressiveness = 0.5):
    '''! Constructor of random agent.
    
    :param name: Name of the agent.
    :type name: str
    :param aggressiveness: Level of aggressiveness. Determines the probability of attacking until dead. 1 means always attacking until dead when attacking.
    :type aggressiveness: float
    '''
    super().__init__(name)
    self.aggressiveness = aggressiveness
    
  
  def pickCountry(self):
    '''! Pick at random one of the empty countries
    '''    
    options = self.board.countriesLeft()
    if options:
      return np.random.choice(options)
    else:
      return None
  
  def placeInitialArmies(self, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''  
    countries  = self.board.getCountriesPlayerWithEnemyNeighbors(self.code)
    if not countries:
      countries = self.board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      # if self.console_debug: print(f"{self.name()}: Placing 1 armies in {c.id}")
      self.board.placeArmies(1, c)
  
  def cardPhase(self, cards):
    '''! Only cash when forced, then cash best possible set
    '''
    if len(cards)<5 or not pyRisk.Deck.containsSet(cards): 
      return None
    c = pyRisk.Deck.yieldBestCashableSet(cards, self.code, self.board.world.countries)
    if not c is None:      
      return c
  
  def placeArmies(self, numberOfArmies:int):
    '''! Place armies at random one by one, but on the countries with enemy borders
    '''
    countries = self.board.getCountriesPlayerWithEnemyNeighbors(self.code)
    if len(countries)==0: 
      # Must have won
      countries = self.board.getCountriesPlayer(self.code)
      # player_countries = '-'.join([c.id for c in self.board.getCountriesPlayer(self.code)])
      # df = self.board.countriesPandas()
      # print(df)
      # print(f"Player {self.code}, {self.name()} has no enemy neighbors on his countries:\n{player_countries }")
      
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      self.board.placeArmies(1, c)
  
  def attackPhase(self):
    '''! Attack a random number of times, from random countries, to random targets.
    The till_dead parameter is also set at random using an aggressiveness parameter
    '''  
    
    nbAttacks = np.random.randint(10)
    for _ in range(nbAttacks):
      canAttack = self.board.getCountriesPlayerThatCanAttack(self.code)
      if len(canAttack)==0: return
      source = np.random.choice(canAttack)
      options = self.board.world.getCountriesToAttack(source.code)
      if not options or source.armies <= 1: continue      
      target = np.random.choice(options)      
      tillDead = True if np.random.uniform()<self.aggressiveness else False
      _ = self.board.attack(source.code, target.code, tillDead)
      # if self.console_debug: print(f"{self.name()}: Attacking {source.id} -> {target_c.id}: tillDead {tillDead}: Result {res}")
    
    return 
  
  def fortifyPhase(self):
    '''! For now, no fortification is made
    '''
    if self.console_debug: print(f"{self.name()}: Fortify: Nothing")
    return 
  
class Human(Agent):
  '''! Agent used to represent a human player.
  It is made so that the GUI works fine.
  It has no methods defined because the behaviour is modeled in the GUI, depending on the human actions (pygame events)
  '''
  def __init__(self, name='human'):
    super().__init__(name)
    self.human = True
    
  def message(self):
    return 'I am human'
  
  def version(self):
    return 'Homo sapiens sapiens'


class PeacefulAgent(Agent):

  def __init__(self, name='peace'):
    '''! Constructor of peaceful agent. It does not attack, so it serves as a very easy baseline
    
    :param name: Name of the agent.
    :type name: str
    '''
    super().__init__(name)
    
    
  
  def pickCountry(self):
    '''! Pick at random one of the empty countries
    '''    
    options = self.board.countriesLeft()
    if options:
      return np.random.choice(options)
    else:
      return None
  
  def placeInitialArmies(self, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''  
    
    countries = self.board.getCountriesPlayer(self.code)
    c = countries[0]
    self.board.placeArmies(numberOfArmies, c)
  
  def cardPhase(self, cards):
    '''! Only cash when forced, then cash best possible set
    '''
    if len(cards)<5 or not pyRisk.Deck.containsSet(cards): 
      return None
    c = pyRisk.Deck.yieldBestCashableSet(cards, self.code, self.board.world.countries)
    if not c is None:      
      return c
  
  def placeArmies(self, numberOfArmies:int):
    '''! Place armies at random one by one, but on the countries with enemy borders
    '''
    countries = self.board.getCountriesPlayer(self.code)
    c = countries[0]
    self.board.placeArmies(numberOfArmies, c)
  
  def attackPhase(self):
    '''! Always peace, never war
    '''  

    return 
  
  def fortifyPhase(self):
    '''! No fortification needed in the path of peace
    '''
    if self.console_debug: print(f"{self.name()}: Fortify: Nothing")
    return 


class Move(object):
  '''! Class used to simpify the idea of legal moves for an agent
  A move is just a tuple (source, target, armies) where wource and target are countries, and armies is an integer.
  In the initial pick, initial fortify and start turn phases, source = target is just the chosen country.
  In the attack and fortify phases, the number of legal moves can be very big if every possible number of armies is considered, so we may limit to some options (1, 5, all) 
  '''
  
  gamePhase_encoder = {'initialPick': '0','initialFortify': '1','startTurn':'2', 'attack':'3', 'fortify':'4'}
  
  def __init__(self, source=None, target=None, details=0, gamePhase = 'unknown'):
    self.source = source
    self.target = target
    self.details = details
    self.gamePhase = Move.gamePhase_encoder[gamePhase]
    
  def encode(self):
    '''! Unique code to represent the move'''
    if self.source is None:
      return self.gamePhase + '_pass'
    else:
      return '_'.join(list(map(str, [self.gamePhase, self.source.code, self.target.code, self.details])))
    
  def __repr__(self):
    if self.source is None:
      return f"{list(Move.gamePhase_encoder.keys())[int(self.gamePhase)]}: pass"
    else:
      return f"{list(Move.gamePhase_encoder.keys())[int(self.gamePhase)]}: {self.source.id} -> {self.target.id}: {self.details}"
  
  @staticmethod
  def buildLegalMoves(board, armies=0):
    '''! Given a board, creates a list of all the legal moves
    Armies is used on the initialFortify and startTurn phases
    '''
    p = board.activePlayer
    if board.gamePhase == 'initialPick':
      return [Move(c,c,0, 'initialPick') for c in board.countriesLeft()]
    elif board.gamePhase in ['initialFortify', 'startTurn']:
      if armies == 0: return []
      return [Move(c,c,a, board.gamePhase) for c,a in itertools.product(board.getCountriesPlayer(p.code), range(armies,armies-1,-1))]
    elif board.gamePhase == 'attack':
      moves = []
      moves.append(Move(None, None, None, 'attack'))
      for source in board.getCountriesPlayerThatCanAttack(p.code):
        for target in board.world.getCountriesToAttack(source.code):
          # Attack once
          moves.append(Move(source, target, 0, 'attack'))
          # Attack till dead
          moves.append(Move(source, target, 1, 'attack'))
      return moves
    elif board.gamePhase == 'fortify':    
      # For the moment, only considering to fortify 5 or all
      moves = []
      moves.append(Move(None, None, None, 'fortify'))
      for source in board.getCountriesPlayer(p.code):
        for target in board.world.getCountriesToFortify(source.code):          
          if source.movable_armies > 0:
            # Fortify all or 1
            moves.append(Move(source, target, 0,'fortify'))            
            moves.append(Move(source, target, 1,'fortify'))
          
          if source.movable_armies > 5:
            moves.append(Move(source, target, 5,'fortify'))
      return moves
      
  @staticmethod
  def play(board, move):
    '''! Simplifies the playing of a move by considering the different game phases
    '''
    if board.gamePhase == 'initialPick':
      board.outsidePickCountry(move.source.code)
    
    elif board.gamePhase in ['initialFortify', 'startTurn']:
      board.outsidePlaceArmies(move.source.code, move.details)
    
    elif board.gamePhase == 'attack':
      if move.source is None: return
      board.attack(move.source.code, move.target.code, bool(move.details))
    
    elif board.gamePhase == 'fortify':   
      if move.source is None: return
      board.fortifyArmies(move.details, move.source.code, move.target.code)
    
                     
#%% Tree search methods

class TreeSearch(Agent):
  '''! Contains methods to generalize the game and be able to perform tree search.
  Tree search algorithms like Flat MC or UCT should be based on this class
  '''
  def __init__(self, name='TreeSearch', playout_policy = RandomAgent):
    super().__init__(name)
    self.playout_policy = playout_policy
    self.move_table = {}    
    
  
  def playout(self, sim_board):
    '''! Simulates a complete game using a policy
    '''    
    sim_board.simulate(self.code, self.playout_policy(), maxRounds=40)
    return sim_board
    
  def score(self, sim_board):
    '''! This functions determines how to score a board.
    One option is to give M>0 if player won, 0 otherwise. But if games are not finished, what would we do?
    Maybe use the number or armies? Number of countries?
    
    Returns a number
    '''
    
    # Very simple win/lose reward, but giving some reward if at least player was still alive
    if sim_board.players[self.code].is_alive:
      if sim_board.getNumberOfPlayersLeft()==1:
        return 100000000
      else:
        s = len(sim_board.getCountriesPlayer(self.code))
        income = sim_board.getPlayerIncome(self.code)
        max_income = 0
        for i, p in sim_board.players.items():
          inc = sim_board.getPlayerIncome(i)
          if inc > max_income and p.code != self.code:
            max_income = inc
        s += max(income - max_income, 0)        
        return s 
    else:
      return 0
 

class FlatMC(TreeSearch):

  def __init__(self, name='flat_mc', playout_policy = RandomAgent, budget = 10):        
    super().__init__(name, playout_policy)
    self.budget = budget
    self.inner_placeArmies_budget = budget//3
    
  def run_flat_mc(self, init_board=None, budget=None, armies = 0):
    board = init_board if not init_board is None else self.board
    budget = budget if not budget is None else self.budget
    moves = Move.buildLegalMoves(board, armies)
    bestScore, bestMove = -9999999999, None
    if self.console_debug: 
      print(f"--FlatMC:run_flat_mc: {self.board.gamePhase}\n Trying {len(moves)} moves, budget of {budget}")
    for m in moves: 
      #print('Looking to pick...')
      #print(m.source.id)
      for i in range(max(budget//len(moves), 1)):  
        #print(i,'... ')
        sim_board = copy.deepcopy(board)
        #print('agent:FlatMC: armies before Move.play', sim_board.world.countries[m.source.code].armies)
        # board.report()
        Move.play(sim_board, m)
        #print('agent:FlatMC: armies after Move.play', sim_board.world.countries[m.source.code].armies)        
        # print("Simulation")
        self.playout(sim_board)  
        score = self.score(sim_board)
        # print(f"done: {score}")
        if score > bestScore:
          if self.console_debug: 
            print(f"------ Found best move:\n\t\tscore: {score}\n\t\tmove {m}")
          bestMove = m
          bestScore = score      
      # print(bestScore)
    return bestMove
      

  def pickCountry(self):
    bestMove = self.run_flat_mc()
    return bestMove.source
    
    
  def placeInitialArmies(self,numberOfArmies:int):
    armies_put = 0
    while armies_put < numberOfArmies:
      # May use more budget than self.budget!
      bestMove = self.run_flat_mc(
        budget = self.inner_placeArmies_budget,
        armies=numberOfArmies-armies_put)
      # print(bestMove, armies_put)
      self.board.placeArmies(int(bestMove.details), bestMove.source)
      armies_put += int(bestMove.details)
    return 
    
  
  def cardPhase(self, cards):
    return pyRisk.Deck.yieldCashableSet(cards)

    # # If cash is possible, simulate with and without cash to choose
    # card_set = pyRisk.Deck.yieldCashableSet(cards)
    
    # if card_set is None: return None
    # if not card_set is None and len(cards)>= 5: return card_set
    
    # # Make simulations
    # bestScore, withCash = -9999999999, None
    
    # # With card cash     
    # for i in range(max(self.budget//2, 1)):
    #   sim_board = copy.deepcopy(self.board)
    #   cashed = sim_board.cashCards(*card_set)
    #   # Place the armies       
    #   armies_put = 0
    #   # Will use more budget!
    #   while armies_put < cashed:
    #     bestMove = self.run_flat_mc(init_board=sim_board, budget = self.inner_placeArmies_budget)
    #     sim_board.placeArmies(int(bestMove.details), bestMove.source)
    #     armies_put += int(bestMove.details)
    #   sim_board = self.playout(sim_board)
    #   score = self.score(sim_board)
    #   if score > bestScore:        
    #     bestScore = score
    #     withCash = True
        
    # # Without cash
    # for i in range(max(self.budget//2, 1)):
    #   sim_board = copy.deepcopy(self.board)    
    #   sim_board = self.playout(sim_board)
    #   score = self.score(sim_board)
    #   if score > bestScore:        
    #     bestScore = score
    #     withCash = False

    # if withCash:
    #   return card_set
    # else:
    #   return None
      
    
  
  
  def placeArmies(self, numberOfArmies:int):
    '''! Given a number of armies, place them on the board
    Use self.board.placeArmies(numberArmies:int, country:Country) to place the armies
    
    :param numberOfArmies: The number of armies the agent must place on the board
    :type numberOfArmies: int
    '''
    armies_put = 0
    while armies_put < numberOfArmies:
      bestMove = self.run_flat_mc(budget = self.inner_placeArmies_budget, armies=numberOfArmies-armies_put)
      self.board.placeArmies(int(bestMove.details), bestMove.source)
      armies_put += int(bestMove.details)
    return 
  
  
  def attackPhase(self):
    '''! Call to attack. 
    Can attack till dead or make single rolls.
    Use res = self.board.attack(source_code:int, target_code:int, tillDead:bool) to do the attack
    Every call to board.attack yields a result as follows:
      - 7: Attacker conquered the target country. This will call to moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int) to determine the number of armies to move to the newly conquered territory
      - 13: Defender won. You are left with only one army and can therefore not continue the attack
      - 0: Neither the attacker nor the defender won. If you want to deduce the armies you lost, you can demand the number of armies in your country.
      - -1: There was an error
      
    You can attack multiple times during the attack phase.
    '''
    
    # For the moment we do not consider the move of just passing to fortification phase.
    # For now, attack until defeated
    res = 7
    while res != -1:
      # May use more budget
      bestMove = self.run_flat_mc()
      # print("FLAT MC: Best move found was ", bestMove)
      if bestMove is None or bestMove.source is None: break
      res = self.board.attack(bestMove.source.code, bestMove.target.code, bool(bestMove.details))
      if res == -1: 
        # print("FLAT MC: Error with the attack")
        raise Exception("FLatMC:attackPhase")
    return
    
  def moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int) -> int:
    # Go with the default for now
    return self.board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self):
    # For now, fortify at most 2 times
    for i in range(2):
      bestMove = self.run_flat_mc()
      if bestMove is None or bestMove.source is None: return 
      self.board.fortifyArmies(int(bestMove.details), bestMove.source.code, bestMove.target.code)
    
  
  def name(self):    
    return self.name_string
  

  
all_agents = {'random': RandomAgent, 'human': Human,
              'flatMC':FlatMC, 'peace':PeacefulAgent}
    