# -*- coding: utf-8 -*-
"""!@package pyRisk

The agent module contains the pyRisk players.
"""

import numpy as np
from deck import Deck
from move import Move
import copy
import itertools


# from mcts.py
from board import Board
from world import World, Country, Continent
from move import Move
import torch
from model import boardToData, buildGlobalFeature
import torch_geometric


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
    :param aggressiveness: Level of aggressiveness. Determines the probability
    of attacking until dead. 1 means always attacking until dead when
    attacking.
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
    '''! Place armies at random one by one, but on the countries with enemy
    borders
    '''
    
    countries = board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      board.placeArmies(1, c)
  
  def attackPhase(self, board):
    '''! Attack a random number of times, from random countries, to random
    targets.
    The till_dead parameter is also set at random using an aggressiveness
    parameter
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
    :param aggressiveness: Level of aggressiveness. Determines the
    probability of attacking until dead. 1 means always attacking
    until dead when attacking.
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
    '''! Place armies at random one by one, but on the countries with
    enemy borders
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
    '''! Attack a random number of times, from random countries, to random
    targets.
    The till_dead parameter is also set at random using an aggressiveness
    parameter
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
  It has no methods defined because the behaviour is modeled in the GUI,
  depending on the human actions (pygame events)
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
    '''! Constructor of peaceful agent. It does not attack, so it serves
    as a very easy baseline
    
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
    '''! Place armies at random one by one, but on the countries with
    enemy borders
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
    moves = board.legalMoves()
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
        sim_board.playMove(m)
        
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
  


#%% Neural MCTS

def heuristic_score_players(state):
    all_income = sum([state.getPlayerIncome(i) for i in state.players])
    all_countries = sum([state.getNumberOfCountriesPlayer(i) for i in state.players])
    num = state.getNumberOfPlayers()
    res = [0]*num
    for i in range(num):
        countries = state.getNumberOfCountriesPlayer(i)
        income = state.getPlayerIncome(i)
        res[i] = 0.45 * (income/all_income) + 0.45 * (countries/all_countries)
    
    return res

def score_players(state):
    num = state.getNumberOfPlayers()    
    if state.getNumberOfPlayersLeft()==1:
        res = [1 if state.players[i].is_alive else -1 for i in range(num)]
    else:
        res = heuristic_score_players(state)
    while len(res) < 6:
        res.append(0.0)
    return np.array(res)

def isTerminal(state):
    if state.getNumberOfPlayersLeft()==1: return True
    return False

def validAttack(state, p, i, j):
    return (state.world.countries[i].owner==p and state.world.countries[j].owner!=p
            and state.world.countries[i].armies>1)    

def validFortify(state, p, i, j):    
    return (state.world.countries[i].owner==p and state.world.countries[j].owner==p
            and state.world.countries[i].moveableArmies>0 and state.world.countries[i].armies>1)
    
def validPick(state, i):    
    return state.world.countries[i].owner == -1

def validPlace(state, p, i):
    return state.world.countries[i].owner == p

def maskAndMoves(state, phase, edge_index):
    p = state.activePlayer.code
    if phase == 'attack':        
        mask = [validAttack(state, p, int(i), int(j))  for i, j in zip(edge_index[0], edge_index[1])]
        moves = [("a", int(i), int(j))  for i, j in zip(edge_index[0], edge_index[1])]
        # Last move is pass
        moves.append(("a", -1, -1))
        mask.append(True)
    elif phase == 'fortify':
        mask = [validFortify(state, p, int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1])]
        moves = [("f", int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1])]
        moves.append(("f", -1, -1))
        mask.append(True)
    elif phase == 'initialPick':
        mask = [validPick(state, int(i)) for i in range(len(state.countries()))]
        moves = [("pi", int(i)) for i in range(len(state.countries()))]
    else: # Place armies
        mask = [validPlace(state, p, int(i)) for i in range(len(state.countries()))]
        moves = [("pl", int(i)) for i in range(len(state.countries()))]
    return torch.FloatTensor(mask).unsqueeze(0), moves

def buildMove(state, move):
    """ move uses the simplified definition used in the function maskAndMoves. 
    This functions turns the simplified notation into a real Move object
    """
    phase = state.gamePhase
    if isinstance(move, int) and move == -1:
        if phase in ['attack', 'fortify']:
            return Move(None, None, None, phase)
        else:
            raise Exception("Trying to pass in a phase where it is not possible")
    
    if phase in ['attack', 'fortify'] and (move[1] == -1):
        return Move(None, None, None, phase)

    s = state.world.countries[move[1]]
    if phase in ['attack', 'fortify']:
        try:
            t = state.world.countries[move[2]]
        except Exception as e:
            print(move)
            raise e
    if phase == 'attack':        
        return Move(s, t, 1, phase)
    elif phase == 'fortify':
        return Move(s, t, s.moveableArmies, phase)
    elif phase == 'initialPick':
        return Move(s, s, 0, phase)
    else: # Place armies
        # TODO: Add armies. For now we put everything in one country
        return Move(s, s, 0, phase)
    
    
class MctsApprentice(object):
    def __init__(self, num_MCTS_sims = 1000, temp=1, max_depth=100):   
        self.apprentice = MCTS(apprentice = None,
                               num_MCTS_sims=num_MCTS_sims, max_depth=max_depth)
        self.temp = temp
    def play(self, state):
        prob, R, Q = self.apprentice.getActionProb(state, temp=self.temp)
        return prob, R


class NetApprentice(object):

    def __init__(self, net):
        self.apprentice = net
        
    def play(self, state):
        canon, map_to_orig = state.toCanonical(state.activePlayer.code)        
        batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
        mask, moves = maskAndMoves(canon, canon.gamePhase, batch.edge_index)
        
        if not self.apprentice is None:
            _, _, _, players, misc = canon.toDicts()
            global_x = buildGlobalFeature(players, misc).unsqueeze(0)
            pick, place, attack, fortify, value = self.apprentice.forward(batch, global_x)            
            if canon.gamePhase == 'initialPick':
                policy = pick
            elif canon.gamePhase in ['initialFortify', 'startTurn']:
                policy = place
            elif canon.gamePhase == 'attack':
                policy = attack
            elif canon.gamePhase == 'fortify':
                policy = fortify
        else:
            policy = torch.ones_like(mask) / max(mask.shape)
        policy = policy * mask
        value = value.squeeze()
        cor_value = torch.FloatTensor([value[map_to_orig.get(i)] if not map_to_orig.get(i) is None else 0.0  for i in range(6)])
        return policy, cor_value


class MCTS(object):

    def __init__(self, apprentice, max_depth = 50, sims_per_eval = 1, num_MCTS_sims = 1000,
                 wa = 10, wb = 10, cb = np.sqrt(2)):
        """ apprentice = None is regular MCTS
            apprentice = neural net -> Expert iteration with policy (and value) net
            apprentice = MCTS -> Used for first iteration
        """
        self.apprentice = apprentice
        self.Nsa, self.Ns, self.Ps, self.Rsa, self.Qsa = {}, {}, {}, {}, {}
        self.Vs, self.As = {}, {}
        self.eps, self.max_depth = 1e-8, max_depth   
        self.sims_per_eval, self.num_MCTS_sims = sims_per_eval, num_MCTS_sims
        self.cb, self.wa, self.wb = cb, wa, wb
    
    def search(self, state, depth, use_val = False):
        # print("\n\n-------- SEARCH --------")
        # print(f"depth: {depth}")
        # state.report()

        # Is terminal? return vector of score per player
        if isTerminal(state) or depth>self.max_depth : 
            # print("\n\n-------- TERMINAL --------")
            return score_players(state), score_players(state)

        # Active player is dead, then end turn
        while not state.activePlayer.is_alive: 
            state.endTurn()
            if state.gameOver: return score_players(state), score_players(state)

        s = hash(state)
        # Is leaf?
        if not s in self.Ps:
            canon, map_to_orig = state.toCanonical(state.activePlayer.code)                        
            batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
            mask, moves = maskAndMoves(canon, canon.gamePhase, batch.edge_index)
            
            if not self.apprentice is None:
                policy, value = self.apprentice.play(canon)
            else:
                # No bias, just uniform sampling for the moment
                policy, value = torch.ones_like(mask)/max(mask.shape), torch.zeros((1,6))
                        
            policy = policy * mask
            self.Vs[s], self.As[s] = mask.squeeze(), moves
            self.Ps[s] = policy.squeeze()
            self.Ns[s] = 1

            # Return an evaluation
            v = np.zeros(6)
            for _ in range(self.sims_per_eval):
                sim = copy.deepcopy(state)
                sim.simulate(RandomAgent())
                v += score_players(sim)                
            v /= self.sims_per_eval

            # Fix order of value returned by net
            value = value.squeeze()
            # Apprentice already does this
            # cor_value = torch.FloatTensor([value[map_to_orig.get(i)] if not map_to_orig.get(i) is None else 0.0  for i in range(6)])
            cor_value = value
            return v, cor_value
        
        # Not a leaf, keep going down. Use values for the current player
        p = state.activePlayer.code
        action = -1
        bestScore = -float('inf')
        # print("Valid:")
        # print(self.Vs[s])
        for i, act in enumerate(self.As[s]):
            a = hash(act)
            # print(i, act)
            if self.Vs[s][i]>0.0:
                if (s,a) in self.Rsa:
                    # PUCT formula
                    uct = self.Rsa[(s,a)][p]+ self.cb*np.sqrt(np.log(self.Ns[s]) / max(self.Nsa[(s,a)], self.eps))
                    val = self.wb*self.Qsa[(s,a)] * (use_val) 
                    pol = self.wa*self.Ps[s][i]/(self.Nsa[(s,a)]+1)
                    sc = uct + pol + val[p]
                else:
                    # Unseen action, take it
                    action = act
                    break
                if sc > bestScore:
                    bestScore = sc
                    action = act
            
        if isinstance(action, int) and action == -1:
            print("**** No move?? *****")
            state.report()
            print(self.As[s])
            print(self.Vs[s])


        # print('best: ', action)
        a = hash(action) # Best action in simplified way
        move = buildMove(state, action)
        # Play action, continue search
        # TODO: For now, armies are placed on one country only to simplify the game
        # print(move)
        state.playMove(move)
        v, net_v = self.search(state, depth+1, use_val)
        if isinstance(net_v, torch.Tensor):
            net_v = net_v.detach().numpy()
        if isinstance(v, torch.Tensor):
            v = v.detach().numpy()

        if (s,a) in self.Rsa:
            rsa, qsa, nsa = self.Rsa[(s,a)], self.Qsa[(s,a)], self.Nsa[(s,a)]
            self.Rsa[(s,a)] = (nsa*rsa + v) /(nsa + 1)
            self.Qsa[(s,a)] = (nsa*qsa + net_v) /(nsa + 1)
            self.Nsa[(s,a)] += 1
        
        else:
            self.Rsa[(s,a)] = v
            self.Qsa[(s,a)] = net_v
            self.Nsa[(s,a)] = 1
        
        self.Ns[s] += 1

        return v, net_v

    """
    def start_search(self, use_val = False, show_final=False):
        state = copy.deepcopy(self.root)
        v, net_v = self.search(state, 0, use_val)
        if show_final:
            print("++++++++++ FINAL ++++++++++")
            state.report()
        return v, net_v
    """

    def getActionProb(self, state, temp=1, num_sims = None, use_val = False, verbose=False):
        """
        This function performs num_MCTS_sims simulations of MCTS starting from
        root
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        if num_sims is None: num_sims = self.num_MCTS_sims
        R, Q = np.zeros(6), np.zeros(6) 
        
        for i in range(num_sims):
            if verbose: progress(i+1, num_sims, f'MCTS:getActionProb:{num_sims}')
            sim = copy.deepcopy(state)
            v, net_v= self.search(sim, 0, use_val)
            R += v
            if isinstance(net_v, np.ndarray):
                Q += net_v
            else:
                Q += net_v.detach().numpy()
        if verbose: print()

        R /= num_sims
        Q /= num_sims

        s = hash(state)
        counts = []

        if not s in self.As:
            # This is happening, but I dont understand why
            state.report()
            print(self.As)
            raise Exception("Looking for state that has not been seen??")


        for i, act in enumerate(self.As[s]):
            a = hash(act)
            counts.append(self.Nsa[(s, a)] if (s, a) in self.Nsa else 0.0)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = torch.FloatTensor([x / counts_sum for x in counts]).view(1,-1)
        return probs, R, Q

class neuralMCTS(Agent):
  '''! MCTS biased by a neural network  
  '''  
  def __init__(self, apprentice, max_depth = 50, sims_per_eval = 1, num_MCTS_sims = 1000,
                 wa = 10, wb = 10, cb = np.sqrt(2), name = "neuralMCTS", move_selection = "argmax", eps_greedy = 0.1):
    '''! Receive the apprentice. 
      None means normal MCTS, it can be a MCTSApprentice or a NetApprentice
      move_selection can be "argmax", "random_proportional" to choose it randomly using the probabilities
      or "e_greddy" to use argmax eps_greedy % of the times, and random_proportional the other % of the times
    '''
    self.name_string = name
    self.MCTS = MCTS(apprentice, max_depth = max_depth, sims_per_eval = sims_per_eval,
                    num_MCTS_sims = num_MCTS_sims,
                    wa = wa, wb = wb, cb = cb)
    self.human = False
    self.console_debug = False
    self.move_selection = move_selection
    self.eps_greedy = eps_greedy
  
  
  def playMove(self, board, temp=1, num_sims=None, use_val = False):
    """ This function will be used in every type of move
        Call the MCTS, get the action probabilities, take the argmax or use any other criterion
    """
    edge_index = boardToData(board).edge_index
    mask, actions = maskAndMoves(board, board.gamePhase, edge_index)
    
    # Do the MCTS
    policy, value, _ = self.MCTS.getActionProb(board, temp=temp, num_sims = num_sims, use_val = use_val)
    
    policy = policy * mask.squeeze().detach().numpy()
    probs = policy / policy.sum()
    
    
    # Use some criterion to choose the move
    z = np.random.uniform()
    if self.move_selection == "argmax" or (self.move_selection == "e_greedy" and z < self.eps_greedy):
      ind = np.argmax(probs)
    elif self.move_selection == "random_proportional" or (self.move_selection == "e_greedy" and z >= self.eps_greedy):
      ind = np.random.choice(range(len(actions)), p = probs)
    
    # Return the selected move
    return buildMove(board, actions[ind])
  
  def pickCountry(self, board) -> int:
    '''! Choose an empty country at the beginning of the game
    '''
    move = self.playMove(board)
    return move.source
    
  def placeInitialArmies(self, board, numberOfArmies:int):
    '''! Place armies in owned countries. No attack phase follows.
    Use board.placeArmies(numberArmies:int, country:Country)
    
    :param numberOfArmies: The number of armies the agent must place on the
    board
    :type numberOfArmies: int
    '''
    move = self.playMove(board)
    # For the moment, all armies are placed in one country.
    # We could use the distribution over the moves to place the armies in that fashion, specially if we use Katago's idea
    # of forced playouts and policy target prunning
    board.placeArmies(numberOfArmies, move.source)
    return 
  
  
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
    if len(cards)<5 or not Deck.containsSet(cards): 
      return None
    c = Deck.yieldBestCashableSet(cards, self.code, board.world.countries)
    if not c is None:      
      return c
  
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Given a number of armies, place them on the board
    Use board.placeArmies(numberArmies:int, country:Country) to place the
    armies
    
    :param numberOfArmies: The number of armies the agent must place on the
    board
    :type numberOfArmies: int
    '''
    move = self.playMove(board)
    # For the moment, all armies are placed in one country.
    # We could use the distribution over the moves to place the armies in that fashion, specially if we use Katago's idea
    # of forced playouts and policy target prunning
    board.placeArmies(numberOfArmies, move.source)
    return 
  
  
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
    move = self.playMove(board)
    res = 0
    while not move.source is None and res != -1:
      res = board.attack(move.source.code, move.target.code, bool(move.details))
      move = self.playMove(board)
    return 
    
  def moveArmiesIn(self, board, countryCodeAttacker:int, countryCodeDefender:int) -> int:
    '''! Default for now
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
    move = self.playMove(board)
    res = 0
    while not move.source is None and res != -1:
      res = board.fortifyArmies(int(move.details), move.source.code, move.target.code)
      move = self.playMove(board)
    return 
    


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
              'flatMC':FlatMC, 'peace':PeacefulAgent, 'neuralMCTS':neuralMCTS}






 
