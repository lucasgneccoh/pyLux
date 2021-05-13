# -*- coding: utf-8 -*-
"""!@package pyRisk

The agent module contains the pyRisk players.
"""

import numpy as np
from pyLux.core.deck import Deck
import copy
import itertools

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
    self.name_string = name
    self.human = False
    self.console_debug = False
  
  def setPrefs(self, code:int):
    '''! Puts the agent into the game by assigning it a code and giving it the reference to the game board
    
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
    
    :param numberOfArmies: The number of armies the agent must place on the board
    :type numberOfArmies: int
    '''
    pass
  
  
  def cardPhase(self, board, cards):
    '''! Call to exchange cards.
    May return None, meaning no cash
    Use :py:module:`pyRisk`.:py:module:`Deck` methods to check for sets and get best sets
    
    :param cards: List with the player's cards
    :type cards: list[:py:module:`pyRisk`.:py:class:`Card`]
    :returns List of three cards to cash
    :rtype list[:py:module:`pyRisk`.:py:class:`Card`]
    '''
    pass
  
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Given a number of armies, place them on the board
    Use board.placeArmies(numberArmies:int, country:Country) to place the armies
    
    :param numberOfArmies: The number of armies the agent must place on the board
    :type numberOfArmies: int
    '''
    pass
  
  
  def attackPhase(self, board):
    '''! Call to attack. 
    Can attack till dead or make single rolls.
    Use res = board.attack(source_code:int, target_code:int, tillDead:bool) to do the attack
    Every call to board.attack yields a result as follows:
      - 7: Attacker conquered the target country. This will call to moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int) to determine the number of armies to move to the newly conquered territory
      - 13: Defender won. You are left with only one army and can therefore not continue the attack
      - 0: Neither the attacker nor the defender won. If you want to deduce the armies you lost, you can demand the number of armies in your country.
      - -1: There was an error
      
    You can attack multiple times during the attack phase.
    '''
    pass
    
  def moveArmiesIn(self, board, countryCodeAttacker:int, countryCodeDefender:int) -> int:
    '''! This method is called when an attack led to a conquer. 
    You can choose the number of armies to send to the conquered territory. 
    The returned value should be between the number of rolled dice and the number of armies in the attacking country minus 1.
    The number of dice rolled is always 3 except when you have 3 or less armies, in which case it will be the number of attacking armies minus 1.
    Notice that by default, this method returns all the moveable armies.
    
    
    :param countryCodeAttacker: Internal code of the attacking country
    :type countryCodeAttacker: int
    :param countryCodeDefender: Internal code of the defending country
    :type countryCodeDefender: int
    :returns Number of armies to move from attacking country to newly conquered territory.
    :rtype int
    '''
    # Default is to move all but one army
    return board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self, board):
    '''! Call to fortify.
    Just before this call, board will update the moveable armies in each territory to the number of armies.
    You can fortify multiple times, always between your countries that are linked and have moveable armies.
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

  def __init__(self, name='random', aggressiveness = 0.1, max_attacks = 10):
    '''! Constructor of random agent.
    
    :param name: Name of the agent.
    :type name: str
    :param aggressiveness: Level of aggressiveness. Determines the probability of attacking until dead. 1 means always attacking until dead when attacking.
    :type aggressiveness: float
    '''
    super().__init__(name)
    self.aggressiveness = aggressiveness
    self.max_attacks = max_attacks
    
  
  def pickCountry(self, board):
    '''! Pick at random one of the empty countries
    '''    
    options = board.countriesLeft()
    if options:
      return np.random.choice(options)
    else:
      return None
  
  def placeInitialArmies(self, board, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''  
    
    countries = board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)      
      board.placeArmies(1, c)
  
  def cardPhase(self, board, cards):
    '''! Only cash when forced, then cash best possible set
    '''
    if len(cards)<5 or not Deck.containsSet(cards): 
      return None
    c = Deck.yieldBestCashableSet(cards, self.code, board.world.countries)
    if not c is None:      
      return c
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Place armies at random one by one, but on the countries with enemy borders
    '''
    
    countries = board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      board.placeArmies(1, c)
  
  def attackPhase(self, board):
    '''! Attack a random number of times, from random countries, to random targets.
    The till_dead parameter is also set at random using an aggressiveness parameter
    '''  
    
    nbAttacks = np.random.randint(self.max_attacks)
    for _ in range(nbAttacks):
      canAttack = board.getCountriesPlayerThatCanAttack(self.code)
      if len(canAttack)==0: return
      source = np.random.choice(canAttack)
      options = board.world.getCountriesToAttack(source.code)
      if not options or source.armies <= 1: continue      
      target = np.random.choice(options)      
      tillDead = True if np.random.uniform()<self.aggressiveness else False
      _ = board.attack(source.code, target.code, tillDead)
      # if self.console_debug: print(f"{self.name()}: Attacking {source.id} -> {target_c.id}: tillDead {tillDead}: Result {res}")
    
    return 
  
  def fortifyPhase(self, board):
    '''! For now, no fortification is made
    '''
    if self.console_debug: print(f"{self.name()}: Fortify: Nothing")
    return 

class RandomAggressiveAgent(RandomAgent):

  def __init__(self, name='randomAggressive', aggressiveness = 0.5):
    '''! Constructor of random agent.
    
    :param name: Name of the agent.
    :type name: str
    :param aggressiveness: Level of aggressiveness. Determines the probability of attacking until dead. 1 means always attacking until dead when attacking.
    :type aggressiveness: float
    '''
    super().__init__(name)
    self.aggressiveness = aggressiveness
  
  def placeInitialArmies(self, board, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''  
    countries  = board.getCountriesPlayerWithEnemyNeighbors(self.code)
    if not countries:
      countries = board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)      
      board.placeArmies(1, c)
  
  def cardPhase(self, board, cards):
    '''! Only cash when forced, then cash best possible set
    '''
    if len(cards)<5 or not Deck.containsSet(cards): 
      return None
    c = Deck.yieldBestCashableSet(cards, self.code, board.world.countries)
    if not c is None:      
      return c
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Place armies at random one by one, but on the countries with enemy borders
    '''
    countries = board.getCountriesPlayerWithEnemyNeighbors(self.code)
    if len(countries)==0: 
      # Must have won
      countries = board.getCountriesPlayer(self.code)
      # player_countries = '-'.join([c.id for c in board.getCountriesPlayer(self.code)])
      # df = board.countriesPandas()
      # print(df)
      # print(f"Player {self.code}, {self.name()} has no enemy neighbors on his countries:\n{player_countries }")
      
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      board.placeArmies(1, c)
  
  def attackPhase(self, board):
    '''! Attack a random number of times, from random countries, to random targets.
    The till_dead parameter is also set at random using an aggressiveness parameter
    '''  
    
    nbAttacks = np.random.randint(10)
    for _ in range(nbAttacks):
      canAttack = board.getCountriesPlayerThatCanAttack(self.code)
      if len(canAttack)==0: return
      source = np.random.choice(canAttack)
      options = board.world.getCountriesToAttack(source.code)
      if not options or source.armies <= 1: continue      
      target = np.random.choice(options)      
      tillDead = True if np.random.uniform()<self.aggressiveness else False
      _ = board.attack(source.code, target.code, tillDead)
      # if self.console_debug: print(f"{self.name()}: Attacking {source.id} -> {target_c.id}: tillDead {tillDead}: Result {res}")
    
    return 
  
  def fortifyPhase(self, board):
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


class PeacefulAgent(RandomAgent):

  def __init__(self, name='peace'):
    '''! Constructor of peaceful agent. It does not attack, so it serves as a very easy baseline
    
    :param name: Name of the agent.
    :type name: str
    '''
    super().__init__(name)
    
  
  def placeInitialArmies(self, board, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''  
    
    countries = board.getCountriesPlayer(self.code)
    c = countries[0]
    board.placeArmies(numberOfArmies, c)
  
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Place armies at random one by one, but on the countries with enemy borders
    '''
    countries = board.getCountriesPlayer(self.code)
    c = countries[0]
    board.placeArmies(numberOfArmies, c)
  
  def attackPhase(self, board):
    '''! Always peace, never war
    '''  

    return 
  
  def fortifyPhase(self, board):
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
  
  
  def __init__(self, source=None, target=None, details=0, gamePhase = 'unknown'):
    self.source = source
    self.target = target
    self.details = details
    self.gamePhase = gamePhase
    
  def encode(self):
    '''! Unique code to represent the move'''
    if self.source is None:
      return self.gamePhase + '_pass'
    else:
      return '_'.join(list(map(str, [self.gamePhase, self.source.code, self.target.code, self.details])))

  def __hash__(self):
    return hash(self.encode())
    
  def __repr__(self):
    return self.encode()
  
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
      for source in board.getCountriesPlayerThatCanAttack(p.code):
        for target in board.world.getCountriesToAttack(source.code):
          # Attack once
          moves.append(Move(source, target, 0, 'attack'))
          # Attack till dead
          moves.append(Move(source, target, 1, 'attack'))
      moves.append(Move(None, None, None, 'attack'))
      return moves
    elif board.gamePhase == 'fortify':    
      # For the moment, only considering to fortify 5 or all
      moves = []
      for source in board.getCountriesPlayer(p.code):
        for target in board.world.getCountriesToFortify(source.code):          
          if source.moveableArmies > 0:
            # Fortify all or 1
            moves.append(Move(source, target, 0,'fortify'))            
            # moves.append(Move(source, target, 1,'fortify'))
          
          if source.moveableArmies > 5:
            moves.append(Move(source, target, 5,'fortify'))
      moves.append(Move(None, None, None, 'fortify'))
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
      try:
        board.attack(move.source.code, move.target.code, bool(move.details))
      except Exception as e:
        raise e
    
    elif board.gamePhase == 'fortify':   
      if move.source is None: return
      board.fortifyArmies(move.details, move.source.code, move.target.code)
    
                     
#%% Tree search methods

class TreeSearch(Agent):
  '''! Contains methods to generalize the game and be able to perform tree search.
  '''
  def __init__(self, name='TreeSearch', playout_policy = RandomAgent):
    super().__init__(name)
    self.playout_policy = playout_policy
    # Dict: [Board (state)][Move (action)][Board (state)]
    # Given a board, we have a set of possible moves
    # For each move, there may be 1 or more states
    # The only case where there are more moves is when attacking
    # In any other case, the after state and action, there will be only one entry
    # For the first part (state), there will be a tuple of things:
    #   (N visits, 
    self.move_table = {}

  def isTerminal(self, board, depth):
    ''' This function determines when to stop the search and use the
        scoring function.        
        Default: Game is over
    '''
    return board.getNumberOfPlayersLeft()==1

  def selectAction(self, board, depth):
    ''' Given a state, chose one of the children'''
    moves = self.move_table[hash(board)]
    return np.random.choice(moves)

  def doRollout(self, board, depth):
    ''' Given a leaf node 
    '''
    return 
  
  def playout(self, sim_board, changeAllAgents = False, maxRounds = 60,
              safety = 10e5, sim_console_debug = False):
    '''! Simulates a complete game using a policy
    '''    
    sim_board.simulate(self.playout_policy(), self.code, changeAllAgents,
                       maxRounds, safety, sim_console_debug)
    return sim_board
    
  def score(self, sim_board):
    '''! This functions determines how to score a board that is considered
    to be terminal.
    Notice that in the common cases, terminal means game is over.
    It can also be used to simulate other
    '''
    # Very simple win/lose reward, but giving some reward if at least
    # player was still alive
    if sim_board.players[self.code].is_alive:
      if sim_board.getNumberOfPlayersLeft()==1:
        # Is the winner
        return 1

      # Game still going
      s = 0
      countries = sim_board.getNumberOfCountriesPlayer(self.code)
      income = sim_board.getPlayerIncome(self.code)
      max_income = max([sim_board.getPlayerIncome(i) for i in sim_board.players])
      max_countries = max([sim_board.getNumberOfCountriesPlayer(i) for i in sim_board.players])
          
      s += 0.45* (income/max_income)
      s += 0.45* (countries/max_countries)
      # Not 1 in the best case to differentiate from actual winning
      return s
    
    else:
      return 0

  def search(self, board, depth):
    return
 

class FlatMC(TreeSearch):

  def __init__(self, name='flat_mc', playout_policy = RandomAgent, budget = 300):        
    super().__init__(name, playout_policy)
    self.budget = budget
    self.inner_placeArmies_budget = budget//3
    
  def run_flat_mc(self, board, budget=None, armies = 0):
    budget = budget if not budget is None else self.budget
    moves = Move.buildLegalMoves(board, armies)
    bestScore, bestMove = -9999999999, None
    if self.console_debug: 
      print(f"--FlatMC:run_flat_mc: {board.gamePhase}\n Trying {len(moves)} moves, budget of {budget}")
    for m in moves: 
      total_reward = 0
      N = max(budget//len(moves), 1)
      for i in range(N):        
        sim_board = copy.deepcopy(board)
        sim_board.replacePlayer(self.code, self.playout_policy())
        
        sim_board.console_debug = False
        Move.play(sim_board, m)
        
        self.playout(sim_board)  
        total_reward += self.score(sim_board)
        
      score = total_reward/N
      if score > bestScore:
        if self.console_debug: 
          print(f"------ Found best move:\n\t\tscore: {score}\n\t\tmove {m}")
        bestMove = m
        bestScore = score
      
    return bestMove
      

  def pickCountry(self, board):
    bestMove = self.run_flat_mc(board)
    return bestMove.source
    
    
  def placeInitialArmies(self, board, numberOfArmies:int):
    armies_put = 0
    while armies_put < numberOfArmies:
      # May use more budget than self.budget!
      bestMove = self.run_flat_mc( board,
        budget = self.inner_placeArmies_budget,
        armies=numberOfArmies-armies_put)
      # print(bestMove, armies_put)
      board.placeArmies(int(bestMove.details), bestMove.source)
      armies_put += int(bestMove.details)
    return 
    
  
  def cardPhase(self, board, cards):
    return Deck.yieldCashableSet(cards)

    # # If cash is possible, simulate with and without cash to choose
    # card_set = pyRisk.Deck.yieldCashableSet(cards)
    
    # if card_set is None: return None
    # if not card_set is None and len(cards)>= 5: return card_set
    
    # # Make simulations
    # bestScore, withCash = -9999999999, None
    
    # # With card cash     
    # for i in range(max(self.budget//2, 1)):
    #   sim_board = copy.deepcopy(board)
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
    #   sim_board = copy.deepcopy(board)    
    #   sim_board = self.playout(sim_board)
    #   score = self.score(sim_board)
    #   if score > bestScore:        
    #     bestScore = score
    #     withCash = False

    # if withCash:
    #   return card_set
    # else:
    #   return None
      
    
  
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Given a number of armies, place them on the board
    Use board.placeArmies(numberArmies:int, country:Country) to place the armies
    
    :param numberOfArmies: The number of armies the agent must place on the board
    :type numberOfArmies: int
    '''
    armies_put = 0
    changed_phase, orig_phase = False, board.gamePhase
    if board.gamePhase == 'attack':      
      # A card cash occured. Can place the armies at any territory
      # To call run_flat_mc, we need to change the gamePhase
      board.gamePhase = 'startTurn'
      changed_phase = True
      
    while armies_put < numberOfArmies:      
      bestMove = self.run_flat_mc(board,
                                  budget = self.inner_placeArmies_budget,
                                  armies=numberOfArmies-armies_put)
      if bestMove is None or bestMove.source is None:
        raise Exception("FlatMC: Nowhere to place armies?. Possibly a call to placeArmies outside of the startTurn phase")        
      board.placeArmies(int(bestMove.details), bestMove.source)
      armies_put += int(bestMove.details)
  
    if changed_phase:
      board.gamePhase = orig_phase
      
    return 
  
  
  def attackPhase(self, board):
    '''! Call to attack. 
    Can attack till dead or make single rolls.
    Use res = board.attack(source_code:int, target_code:int, tillDead:bool) to do the attack
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
      bestMove = self.run_flat_mc(board)
      # print("FLAT MC: Best move found was ", bestMove)
      if bestMove is None or bestMove.source is None: break
      res = board.attack(bestMove.source.code, bestMove.target.code, bool(bestMove.details))
      if res == -1: 
        # print("FLAT MC: Error with the attack")
        raise Exception("FLatMC:attackPhase")
    return
    
  def moveArmiesIn(self, board, countryCodeAttacker:int, countryCodeDefender:int) -> int:
    # Go with the default for now
    return board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self, board):
    # For now, fortify at most 1 time
    for i in range(1):
      bestMove = self.run_flat_mc(board)
      if bestMove is None or bestMove.source is None: return 
      board.fortifyArmies(int(bestMove.details), bestMove.source.code, bestMove.target.code)
    
  
  def name(self):    
    return self.name_string
  

'''
For the python player in Java, the representation of the board is much more basic.
Instead of having the pyRisk board with the world object and all its methods, 
the board that comes from Java is represented using:
  - List of countries with code, name, owner, continent, armies, moveableArmies
  - List of continents with code, name, bonus
  - List of incoming edges for each node.
  - List of players with player code, name, income, number of cards
  
  use Board.fromDicts and Board.toDicts to handle such a representation
'''




  
all_agents = {'random': RandomAgent, 'random_aggressive':RandomAggressiveAgent,
              'human': Human,
              'flatMC':FlatMC, 'peace':PeacefulAgent}
    
