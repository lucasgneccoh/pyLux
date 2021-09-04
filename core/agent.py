# -*- coding: utf-8 -*-
"""!@package pyRisk

The agent module contains the pyRisk players.
"""

import numpy as np
from deck import Deck
from move import Move
import misc

import copy
import itertools
import os

from board import Board, Agent, RandomAgent
from world import World

import torch

# Needs torch_geometric
from model import boardToData, buildGlobalFeature, GCN_risk, load_dict
import torch_geometric


class HumanAgent(RandomAgent):

  def __init__(self, name='human'):
    super().__init__(name = name)
    self.human = True
    


class PeacefulAgent(RandomAgent):

  def __init__(self, name='peace'):
    '''! Constructor of peaceful agent. It does not attack, so it serves
    as a very easy baseline
    
    :param name: Name of the agent.
    :type name: str
    '''
    super().__init__(name = name)
    
  
  def placeInitialArmies(self, board, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''      
    countries = board.getCountriesPlayer(self.code)
    c = countries[0]
    return Move(c, c, numberOfArmies, board.gamePhase)
  
  
  def placeArmies(self, board, numberOfArmies:int):
    '''! Place armies at random one by one, but on the countries with
    enemy borders
    '''
    countries = board.getCountriesPlayer(self.code)
    c = countries[0]
    return Move(c, c, numberOfArmies, board.gamePhase)
  
  def attackPhase(self, board):
    '''! Always peace, never war
    '''  

    return Move(None, None, None, board.gamePhase)
  
  def fortifyPhase(self, board):
    '''! No fortification needed in the path of peace
    '''
    return Move(None, None, None, board.gamePhase)
    

# Helper functions for MCTS
def heuristic_score_players(state):
    all_income = sum([state.getPlayerIncome(i) for i in state.players])
    all_countries = sum([state.getNumberOfCountriesPlayer(i) for i in state.players])
    num = state.getNumberOfPlayers()
    res = [0]*num
    for i in range(num):
        countries = state.getNumberOfCountriesPlayer(i)
        income = state.getPlayerIncome(i)
        res[i] = 0.5 * (income/all_income) + 0.5 * (countries/all_countries)
    
    return res

def score_players(state):
    num = state.getNumberOfPlayers()    
    if state.getNumberOfPlayersLeft()==1:
        res = [1 if state.players[i].is_alive else 0 for i in range(num)]
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
    # Needed this representation of moves for the neural network. board.legalMoves is not well suited
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


class FlatMC(object):
    """ Flat MC
    """
    def __init__(self, max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000,
                  cb = 0):
        
        # Ps is the policy in state s, Rsa is the reward, Qsa the estimated value (using the net for example)
        self.Nsa, self.Ns, self.Ps, self.Rsa, self.Qsa = {}, {}, {}, {}, {}
        # Valid moves (mask) and list of moves
        self.Vs, self.As = {}, {}
        self.eps, self.max_depth = 1e-8, max_depth   
        self.sims_per_eval, self.num_MCTS_sims = sims_per_eval, num_MCTS_sims
        self.cb = cb
        self.cycle = None
        self.root = None
    
    def onLeaf(self, state, depth):
        #print("onLeaf")
        s = hash(state)
        # Expand the node only if it is the root
        if depth == 0:
            self.root = s
            # Expand 
            moves = state.legalMoves()
            policy, _ = np.zeros(len(moves)), np.zeros(6) 
            # All moves are legal because we called board.legalMoves
            self.Vs[s] = np.ones(len(moves))
            self.As[s] = moves
            self.Ps[s] = policy
            self.Ns[s] = 1
            
        # Make a playout in any case        
        # Return an evaluation
        v = np.zeros(6)
        for _ in range(self.sims_per_eval):
            sim = copy.deepcopy(state)
            sim.simulate(sim_console_debug = False)
            v += score_players(sim)                
        v /= self.sims_per_eval
        
        #print("onLeaf: End")
        return v, v
        
    def treePolicy(self, state):
        #print("Tree policy")
        # Try all moves in a cycle
        if self.cycle is None:
            self.cycle = itertools.cycle(self.As[self.root])
        
        # Next action is just the following one to test
        return next(self.cycle)
        
    
    def search(self, state, depth):
        #print("\n\n-------- SEARCH --------")
        #print(f"depth: {depth}")
        # state.report()

        # Is terminal? return vector of score per player
        if isTerminal(state) or depth>self.max_depth : 
            #print("\n\n-------- TERMINAL --------")            
            return score_players(state), score_players(state)

        # Active player is dead, then end turn
        while not state.activePlayer.is_alive: 
            state.endTurn()
            # Just in case
            if state.gameOver: return score_players(state), score_players(state)

        s = hash(state)
        # Is leaf?
        if not s in self.Ps:
            v, cor_value = self.onLeaf(state, depth)
            return v, cor_value
            
        # Not a leaf, keep going down. Use values for the current player
        action = self.treePolicy(state)
        
        if isinstance(action, int) and action == -1:
            print("**** No move?? *****")
            state.report()
            print(self.As[s])
            print(self.Vs[s])


        #print('best: ', action)
        a = hash(action) # Best action in simplified way
        if not isinstance(action, Move):
          move = buildMove(state, action)
        else:
          move = action
        # Play action, continue search
        # TODO: For now, armies are placed on one country only to simplify the game
        #print(move)
        
        state.playMove(move)
        
        # Once the search is done, update values for current (state, action) using the hashes s and a
        try:
            v, net_v = self.search(state, depth+1)
        except Exception as e:
            print("Problems while doing search")
            print(e)
        
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


    def getBestAction(self, state, player, num_sims = None, verbose=False):
        """
        This function performs num_MCTS_sims simulations of MCTS starting from
        state
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        #state = copy.deepcopy(board)
        #state.readyForSimulation()
        #state.console_debug = False
        
        if num_sims is None: num_sims = self.num_MCTS_sims
        R, Q = np.zeros(6), np.zeros(6) 
        
        for i in range(num_sims):
            if verbose: misc.print_message_over(f'FlatMC:getBestAction:{i+1} / {num_sims}')            
            sim = copy.deepcopy(state)            
            #sim.console_debug = False
            v, net_v= self.search(sim, 0)
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

        bestAction = None
        bestValue = -float('inf')
        for i, act in enumerate(self.As[s]):
            a = hash(act)
            if (s,a) in self.Rsa:
                if self.Rsa[(s,a)][player] >= bestValue:
                    bestValue = self.Rsa[(s,a)][player]
                    bestAction = act
            else:
                pass
                
        return bestAction, bestValue, R, Q
            

    def getVisitCount(self, state, temp=1):
        s = hash(state)
        if not s in self.As:
            return None
        counts = []
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
        probs = [x / counts_sum for x in counts]
        return np.array(probs)

class UCT(object):
    """ UCT
    """
    def __init__(self, max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000,
                  cb = np.sqrt(2)):
        
        # Ps is the policy in state s, Rsa is the reward, Qsa the estimated value (using the net for example)
        self.Nsa, self.Ns, self.Ps, self.Rsa, self.Qsa = {}, {}, {}, {}, {}
        # Valid moves (mask) and list of moves
        self.Vs, self.As = {}, {}
        self.eps, self.max_depth = 1e-8, max_depth   
        self.sims_per_eval, self.num_MCTS_sims = sims_per_eval, num_MCTS_sims
        self.cb = cb
        self.cycle = None
        self.root = None
    
    def onLeaf(self, state, depth):
        #print("onLeaf")
        s = hash(state)

        
        # Expand (no networks used, just to call get mask and be aligned with PUCT)
        batch = torch_geometric.data.Batch.from_data_list([boardToData(state)])
        mask, moves = maskAndMoves(state, state.gamePhase, batch.edge_index)
                
        policy, _ = np.zeros(len(moves)), np.zeros(6) 
        # All moves are legal because we called board.legalMoves
        self.Vs[s] = mask.squeeze().detach().numpy()
        self.As[s] = moves
        self.Ps[s] = policy
        self.Ns[s] = 1
        
        # Make a playout in any case        
        # Return an evaluation
        v = np.zeros(6)
        for _ in range(self.sims_per_eval):
            sim = copy.deepcopy(state)
            sim.simulate(sim_console_debug = False)
            v += score_players(sim)                
        v /= self.sims_per_eval
        
        return v, v
        
    def treePolicy(self, state):
        s = hash(state)
        p = state.activePlayer.code
        action = None
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
                    val = 0 # self.wb*self.Qsa[(s,a)] * (use_val) 
                    pol = 0 #self.wa*self.Ps[s][i]/(self.Nsa[(s,a)]+1)
                    sc = uct + val + pol
                else:
                    # Unseen action, take it
                    action = act
                    break
                if sc > bestScore:
                    bestScore = sc
                    action = act
        return action
        
    
    def search(self, state, depth):
        #print("\n\n-------- SEARCH --------")
        #print(f"depth: {depth}")
        # state.report()

        # Is terminal? return vector of score per player
        if isTerminal(state) or depth>self.max_depth : 
            #print("\n\n-------- TERMINAL --------")            
            return score_players(state), score_players(state)

        # Active player is dead, then end turn
        while not state.activePlayer.is_alive: 
            state.endTurn()
            # Just in case
            if state.gameOver: return score_players(state), score_players(state)

        s = hash(state)
        # Is leaf?
        if not s in self.Ps:
            v, cor_value = self.onLeaf(state, depth)
            return v, cor_value
            
        # Not a leaf, keep going down. Use values for the current player
        action = self.treePolicy(state)
        
        if isinstance(action, int) and action == -1:
            print("**** No move?? *****")
            state.report()
            print(self.As[s])
            print(self.Vs[s])


        #print('best: ', action)
        a = hash(action) # Best action in simplified way
        if not isinstance(action, Move):
          move = buildMove(state, action)
        else:
          move = action
        # Play action, continue search
        # TODO: For now, armies are placed on one country only to simplify the game
        #print(move)
        state.playMove(move)
        
        # Once the search is done, update values for current (state, action) using the hashes s and a
        try:
            v, net_v = self.search(state, depth+1)
        except Exception as e:
            print("Problems while doing search")
            print(e)
            # Define v and net_v??
        
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


    def getBestAction(self, state, player, num_sims = None, verbose=False):
        """
        This function performs num_MCTS_sims simulations of MCTS starting from
        state
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """        
        
        if num_sims is None: num_sims = self.num_MCTS_sims
        R, Q = np.zeros(6), np.zeros(6) 
        
        for i in range(num_sims):
            if verbose: misc.print_message_over(f'UCT:getBestAction:{i+1} / {num_sims}')            
            sim = copy.deepcopy(state)
            sim.console_debug = False
            v, net_v= self.search(sim, 0)
            R += v
            if isinstance(net_v, np.ndarray):
                Q += net_v
            else:
                Q += net_v.detach().numpy()
        if verbose: print()

        R /= num_sims
        Q /= num_sims

        s = hash(state)
        

        if not s in self.As:
            # This is happening, but I dont understand why
            state.report()
            print(self.As)
            raise Exception("Looking for state that has not been seen??")

        bestAction = None
        bestValue = -float('inf')
        for i, act in enumerate(self.As[s]):
            a = hash(act)
            if (s,a) in self.Rsa:
                if self.Rsa[(s,a)][player] >= bestValue:
                    bestValue = self.Rsa[(s,a)][player]
                    bestAction = act
            else:
                pass
        
        
        return bestAction, bestValue, R, Q


    def getVisitCount(self, state, temp=1):
        s = hash(state)
        if not s in self.As:
            return None
            
        counts = []
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
        probs = [x / counts_sum for x in counts]        
        return np.array(probs)
        

class FlatMCPlayer(Agent):

    def __init__(self, name='flatMC', max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000,
                  cb = 0):        
        self.name = name
        self.human = False
        self.console_debug = False  
        self.max_depth = max_depth
        self.sims_per_eval = sims_per_eval
        self.num_MCTS_sims = num_MCTS_sims
        self.cb = cb
        self.flat = FlatMC(max_depth, sims_per_eval, num_MCTS_sims, cb)
        
 
    def run(self, board, num_sims=None): 
        self.flat = FlatMC(self.max_depth, self.sims_per_eval, self.num_MCTS_sims, self.cb)
        
        state = copy.deepcopy(board)
        state.readyForSimulation()
        state.console_debug = False
        
        bestAction, bestValue, _, _ = self.flat.getBestAction(state, self.code, num_sims = num_sims, verbose=False)
                
        return bestAction
      
    
    def pickCountry(self, board):
        '''! Pick at random one of the empty countries
        '''            
        move = self.run(board) 
        return move
      
    
    def placeInitialArmies(self, board, numberOfArmies:int):
        '''! Pick at random one of the empty countries
        '''  
        
        return self.run(board)
      
      
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
        return self.run(board)
    
    def attackPhase(self, board):
        '''! Attack a random number of times, from random countries, to random
        targets.
        The till_dead parameter is also set at random using an aggressiveness
        parameter
        '''  
        return self.run(board)
    
    def fortifyPhase(self, board):
        '''! For now, no fortification is made
        '''
        return self.run(board)
        

class UCTPlayer(Agent):

    def __init__(self, name='UCT', max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000,
                  cb = np.sqrt(2), verbose = False):        
        self.name = name
        self.human = False
        self.console_debug = False 
        self.max_depth = max_depth
        self.sims_per_eval = sims_per_eval
        self.num_MCTS_sims = num_MCTS_sims
        self.cb = cb
        self.uct = UCT(max_depth, sims_per_eval, num_MCTS_sims, cb)
        self.verbose = verbose
        
 
    def run(self, board, num_sims=None): 
        state = copy.deepcopy(board)
        state.readyForSimulation()
        state.console_debug = False        
        self.uct = UCT(self.max_depth, self.sims_per_eval, self.num_MCTS_sims, self.cb)    
        bestAction, bestValue, _, _ = self.uct.getBestAction(state, self.code, num_sims = num_sims, verbose=self.verbose)
        return buildMove(board, bestAction)
      
    
    def pickCountry(self, board):
        '''! Pick at random one of the empty countries
        '''    
        return self.run(board)
      
    
    def placeInitialArmies(self, board, numberOfArmies:int):
        '''! Pick at random one of the empty countries
        '''  
        return self.run(board)
      
      
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
        return self.run(board)
    
    def attackPhase(self, board):
        '''! Attack a random number of times, from random countries, to random
        targets.
        The till_dead parameter is also set at random using an aggressiveness
        parameter
        '''  
        return self.run(board)
    
    def fortifyPhase(self, board):
        '''! For now, no fortification is made
        '''
        return self.run(board)


class MctsApprentice(object):
    def __init__(self, num_MCTS_sims = 1000, sims_per_eval = 1, temp=1, max_depth=100): 
        self.max_depth = max_depth
        self.num_MCTS_sims = num_MCTS_sims        
        self.temp = temp
        self.sims_per_eval = sims_per_eval
        self.apprentice = UCT(num_MCTS_sims=num_MCTS_sims, max_depth=max_depth, sims_per_eval = sims_per_eval)
        
    def play(self, state):
        # Restart tree
        self.apprentice = UCT(num_MCTS_sims=self.num_MCTS_sims, max_depth=self.max_depth, sims_per_eval = self.sims_per_eval)
        bestAction, bestValue, _, _ = self.apprentice.getBestAction(state, state.activePlayer.code)
        return bestAction, bestValue
    
    def getPolicy(self, state):
        self.apprentice = UCT(num_MCTS_sims=self.num_MCTS_sims, max_depth=self.max_depth)
        bestAction, bestValue, R, Q = self.apprentice.getBestAction(state, state.activePlayer.code)
        #print("\n\n************* MctsApprentice: statistics of root node\n\n")
        #print(self.apprentice.As[hash(state)])
        #print(self.apprentice.Ns[hash(state)])
        #for i, a in enumerate(self.apprentice.As[hash(state)]):
        #    if self.apprentice.Vs[hash(state)][i]>0: print(a, " -- ", self.apprentice.Nsa[(hash(state), hash(a))])
        
        probs = self.apprentice.getVisitCount(state, temp = self.temp)
        return probs, R
        

class NetApprentice(object):

    def __init__(self, net):
        self.apprentice = net
        
    def getPolicy(self, canon):     
        # state must be already in canonical form. The correction must be done outside
        batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
        mask, moves = maskAndMoves(canon, canon.gamePhase, batch.edge_index)
        
        if not self.apprentice is None:
            # Get information relevant for global features
            _, _, _, players, misc = canon.toDicts()
            # Build global features
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
        # value is given in canonical order, must reorder to original player order
        return policy.detach().numpy(), value.detach().numpy()
        
    def play(self, canon, play_mode = "argmax"):
        policy, value = self.getPolicy(canon)
        batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
        mask, moves = maskAndMoves(canon, canon.gamePhase, batch.edge_index)
        ind = np.argmax(policy.detach().numpy())
        return moves[ind], -1 



class PUCT(object):

    # TODO: Add the minimum number of visits and the prunning on the getBestAction (Katago) 

    def __init__(self, apprentice, max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000,
                 wa = 10, wb = 10, cb = np.sqrt(2), use_val = 0, console_debug = False):
        """ apprentice = None is regular MCTS
            apprentice = neural net -> Expert iteration with policy (and value) net
            apprentice = MCTS -> Used for first iteration
        """
        self.apprentice = apprentice
        # Ps is the policy in state s, Rsa is the reward, Qsa the estimated value (using the net for example)
        self.Nsa, self.Ns, self.Ps, self.Rsa, self.Qsa = {}, {}, {}, {}, {}
        # Valid moves (mask) and list of moves
        self.Vs, self.As = {}, {}
        self.eps, self.max_depth = 1e-8, max_depth   
        self.sims_per_eval, self.num_MCTS_sims = sims_per_eval, num_MCTS_sims
        self.cb, self.wa, self.wb = cb, wa, wb
        self.use_val = use_val
        self.console_debug = console_debug
    
    def onLeaf(self, state, depth):
        s = hash(state)
        # Expand the node 
        canon, map_to_orig = state.toCanonical(state.activePlayer.code)                        
        batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
        mask, moves = maskAndMoves(canon, canon.gamePhase, batch.edge_index)
        
        if not self.apprentice is None:
            policy, net_value = self.apprentice.getPolicy(canon)
            # Fix order of value returned by net
            net_value = net_value.squeeze()
            # Apprentice already does this
            cor_value = np.array([net_value[map_to_orig.get(i)] if not map_to_orig.get(i) is None else 0.0  for i in range(6)])
        else:
            raise Exception("PUCT without apprentice")
                    
        
        #if self.console_debug: print(f"OnLeaf: State {state.board_id} ({s})")
        #if self.console_debug: print(f"OnLeaf: Found this actions to expand {moves}")
        
        if not isinstance(policy, torch.Tensor):
            policy = torch.Tensor(policy)
            
        policy = policy * mask
        self.Vs[s], self.As[s] = mask.squeeze().detach().numpy(), moves
        self.Ps[s] = policy.squeeze().detach().numpy()
        self.Ns[s] = 1

        # Monte Carlo evaluation       
        v = np.zeros(6)
        for _ in range(self.sims_per_eval):
            sim = copy.deepcopy(state)
            sim.simulate()
            v += score_players(sim)                
        v /= self.sims_per_eval        
        
        # Value MC, value estimated by the network
        return v, cor_value
        
    def treePolicy(self, state):
        s = hash(state)
        p = state.activePlayer.code
        action = -1
        bestScore = -float('inf')
        
        
        #if self.console_debug: print("treePolicy: Start")
        #if self.console_debug: print("Valid:")
        #if self.console_debug: print(self.Vs[s])
        #if self.console_debug: print("Actions:")
        #if self.console_debug: print(self.As[s])
                
        for i, act in enumerate(self.As[s]):
            a = hash(act)
            # print(i, act)
            if self.Vs[s][i]>0.0:
                if (s,a) in self.Rsa:
                    # PUCT formula
             
                    mean = (1-self.use_val)*self.Rsa[(s,a)][p] + self.use_val*self.Qsa[(s,a)][p]
                    prior = self.Ps[s][i]
                    sc = mean + self.cb * prior * np.sqrt(self.Ns[s]) / (self.Nsa[(s,a)]+1)
                    
                    #if self.console_debug: print(f"treePolicy: score for action {act}:  {sc}")
                else:
                    # Unseen action, take it
                    #if self.console_debug: print(f"treePolicy: unseen action {act}")
                    action = act
                    break
                if sc > bestScore:
                    bestScore = sc
                    action = act
        return action
        
    
    def search(self, state, depth):
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
            # Just in case
            if state.gameOver: return score_players(state), score_players(state)

        s = hash(state)
        # Is leaf?
        if not s in self.Ps:
            v, cor_value = self.onLeaf(state, depth)
            return v, cor_value
            
        # Not a leaf, keep going down. Use values for the current player
        action = self.treePolicy(state)
        
        #if self.console_debug: print(f"Best action found by tree policy: {action}")
        
        if isinstance(action, int) and action == -1:
            print("**** No move?? *****")
            state.report()
            print(self.As[s])
            print(self.Vs[s])


        
        a = hash(action) # Best action in simplified way        
        
        # Play action, continue search
        # TODO: For now, armies are placed on one country only to simplify the game
        
        
        if not isinstance(action, Move):
          move = buildMove(state, action)
        else:
          move = action
        
        # if self.console_debug: print(move)
        
        state.playMove(move)
        
        # Once the search is done, update values for current (state, action) using the hashes s and a
        try:
            v, net_v = self.search(state, depth+1)
        except Exception as e:
            print("Problems while doing search")
            print(e)
        
        if isinstance(net_v, torch.Tensor):
            net_v = net_v.detach().numpy()
        if isinstance(v, torch.Tensor):
            v = v.detach().numpy()

        if (s,a) in self.Rsa:
            rsa, nsa, qsa = self.Rsa[(s,a)], self.Nsa[(s,a)], self.Qsa[(s,a)]
            self.Rsa[(s,a)] = (nsa*rsa + v) /(nsa + 1)
            self.Qsa[(s,a)] = (nsa*qsa + net_v) /(nsa + 1)
            self.Nsa[(s,a)] += 1
        
        else:
            self.Rsa[(s,a)] = v
            self.Qsa[(s,a)] = net_v
            self.Nsa[(s,a)] = 1
        
        self.Ns[s] += 1

        return v, net_v


    def getBestAction(self, state, player, num_sims = None, verbose=False):
        """
        This function performs num_MCTS_sims simulations of MCTS starting from
        state
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        if num_sims is None: num_sims = self.num_MCTS_sims
        R, Q = np.zeros(6), np.zeros(6) 
        
        for i in range(num_sims):
            if verbose: misc.print_message_over(f"PUCT:getBestAction: simulation {i}")            
            sim = copy.deepcopy(state)
            v, net_v= self.search(sim, 0)
            R += v
            if isinstance(net_v, np.ndarray):
                Q += net_v
            else:
                Q += net_v.detach().numpy()
        if verbose: print()

        R /= num_sims
        Q /= num_sims

        s = hash(state)

        if not s in self.As:
            # This is happening, but I dont understand why
            state.report()
            print(self.As)
            raise Exception("Looking for state that has not been seen??")


        bestAction = None
        bestValue = -float('inf')
        for i, act in enumerate(self.As[s]):
            a = hash(act)
            if (s,a) in self.Rsa:
                sc = (1-self.use_val)*self.Rsa[(s,a)][player] + self.use_val*self.Qsa[(s,a)][player]
                if sc >= bestValue:
                    bestValue = sc
                    bestAction = act
            else:
                pass
                
        return bestAction, bestValue, R, Q
        
    def getVisitCount(self, state, temp=1):
        s = hash(state)
        if not s in self.As:
            return None
                   
        counts = []
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
        probs = [x / counts_sum for x in counts]
        return np.array(probs)



class PUCTPlayer(Agent):
  """! MCTS biased by a neural network  
  """  
  def __init__(self, apprentice = None, name = "neuralMCTS", max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000,
                 wa = 10, wb = 10, cb = np.sqrt(2), temp = 1, use_val = 0, move_selection = "argmax", console_debug = False):
      """! Receive the apprentice. 
        None means normal MCTS, it can be a MCTSApprentice or a NetApprentice
        move_selection can be "argmax", "random_proportional" to choose it randomly using the probabilities
        
      """
      self.name = name    
      self.human = False
      self.console_debug = console_debug
      
      self.apprentice = apprentice
      self.max_depth = max_depth
      self.sims_per_eval = sims_per_eval
      self.num_MCTS_sims = num_MCTS_sims
      self.wa = wa
      self.wb = wb
      self.cb = cb
      self.temp = temp
      self.use_val = use_val
      self.move_selection = move_selection      
  
  def run(self, board):
      """ This function will be used in every type of move
          Call the MCTS, get the action probabilities, take the argmax or use any other criterion
      """      
      
      # Do the MCTS
      
      # First copy the state and remove player to avoid copying the puct object
      state = copy.deepcopy(board)
      state.readyForSimulation()
      state.console_debug = False
      
      self.puct = PUCT(self.apprentice, self.max_depth, self.sims_per_eval, self.num_MCTS_sims,
                   self.wa, self.wb, self.cb, use_val = self.use_val, console_debug = self.console_debug)
                   
      
      bestAction, bestValue, _, _ = self.puct.getBestAction(state, player = self.code, num_sims = None, verbose = state.console_debug)
      probs = self.puct.getVisitCount(state, temp=self.temp)
      actions = self.puct.As[hash(board)]
      # Use some criterion to choose the move
      if self.move_selection == "argmax":
          ind = np.argmax(probs)
      elif self.move_selection == "random_proportional":
          ind = np.random.choice(range(len(actions)), p = probs)
      else:
          raise Exception("Invalid kind of move selection criterion")
      
      # Return the selected move, destroy tree
      self.puct = None # Dont destroy tree?
      return buildMove(board, actions[ind])
  
  def pickCountry(self, board) -> int:
      move = self.run(board)
      return move
    
  def placeInitialArmies(self, board, numberOfArmies:int):
      move = self.run(board)
      return move
  
  
  def cardPhase(self, board, cards):
      if len(cards)<5 or not Deck.containsSet(cards): 
        return None
      c = Deck.yieldBestCashableSet(cards, self.code, board.world.countries)
      if not c is None:      
        return c
  
  
  def placeArmies(self, board, numberOfArmies:int):    
      move = self.run(board)
      return move
  
  
  def attackPhase(self, board):
      move = self.run(board)
      return move 
    
  def moveArmiesIn(self, board, countryCodeAttacker:int, countryCodeDefender:int) -> int:
      # Default for now
      
      # Default is to move all but one army
      return board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self, board):      
      move = self.run(board)
      return move
  

class NetPlayer(Agent):  
  def __init__(self, apprentice = None, name = "netPlayer", move_selection = "random_proportional", temp = 1):
      
      self.name = name    
      self.human = False
      self.console_debug = False
      
      self.apprentice = apprentice     
      self.move_selection = move_selection  
      self.temp = temp
  
  def run(self, state):
    
      canon, map_to_orig = state.toCanonical(state.activePlayer.code)                        
      batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
      mask, moves = maskAndMoves(canon, canon.gamePhase, batch.edge_index)      
      policy, net_value = self.apprentice.getPolicy(canon)
      # Fix order of value returned by net
      # net_value = net_value.squeeze()
      
      # cor_value = np.array([net_value[map_to_orig.get(i)] if not map_to_orig.get(i) is None else 0.0  for i in range(6)])
      
      pol = (policy * mask.detach().numpy()).squeeze()
      if self.temp == 0: 
          self.move_selection = "argmax"
          T = 1
      else:
          T = self.temp
      
      exp = np.exp(np.log(np.maximum(pol,0.0001))/T)
      S = exp.sum()
      probs = exp/S
      
      # Use some criterion to choose the move
      if self.move_selection == "argmax":
          ind = np.argmax(probs)
      elif self.move_selection == "random_proportional":
          ind = np.random.choice(len(moves), p = probs)
      else:
          raise Exception("Invalid kind of move selection criterion")
      
      print()
      print(f"Net player policy with temp {self.temp}, {self.move_selection}")
      print("Move -- Mask -- Orig pol -- softmax")
      moves_aux = []
      countries = state.countries()
      for m in moves:
          if len(m) > 2:
              moves_aux.append((m[0], countries[m[1]].id, countries[m[2]].id))
          else:
              moves_aux.append((m[0], countries[m[1]].id))
      print(*zip(moves_aux, mask.detach().numpy().squeeze(), pol.round(3), probs.round(3)), sep="\n")
      
      
      return buildMove(state, moves[ind])
  
  def pickCountry(self, board) -> int:
      move = self.run(board)
      return move
    
  def placeInitialArmies(self, board, numberOfArmies:int):
      move = self.run(board)
      return move
  
  
  def cardPhase(self, board, cards):
      if len(cards)<5 or not Deck.containsSet(cards): 
        return None
      c = Deck.yieldBestCashableSet(cards, self.code, board.world.countries)
      if not c is None:      
        return c
  
  
  def placeArmies(self, board, numberOfArmies:int):    
      move = self.run(board)
      return move
  
  
  def attackPhase(self, board):
      move = self.run(board)
      return move 
    
  def moveArmiesIn(self, board, countryCodeAttacker:int, countryCodeDefender:int) -> int:
      # Default for now
      
      # Default is to move all but one army
      return board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self, board):      
      move = self.run(board)
      return move
  


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
  
'''
all_agents = {'random': RandomAgent, 'random_aggressive':RandomAggressiveAgent,
              'human': Human,
              'flatMC':FlatMC, 'peace':PeacefulAgent, 'neuralMCTS':neuralMCTS}
'''


if __name__ == "__main__":
    # Load a board, try to play

    # Load map
    #path = '../support/maps/classic_world_map.json'
    path = '../support/maps/diamond_map.json'
    path = '../support/maps/test_map.json'
      
    world = World(path)
    

    # Set players  
    pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Green'), RandomAgent('Blue')
    
    pFlat = FlatMCPlayer(name='flatMC', max_depth = 300, sims_per_eval = 2, num_MCTS_sims = 700, cb = 0)
    pUCT = UCTPlayer(name='UCT', max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000, cb = np.sqrt(2), verbose = True)
    
    players = [pR2, pR1]
    # Set board
    prefs = {'initialPhase': True, 'useCards':True,
             'transferCards':True, 'immediateCash': True,
             'continentIncrease': 0.05, 'pickInitialCountries':True,
             'armiesPerTurnInitial':4,
             'console_debug':True}  
             
    board_orig = Board(world, players)
    board_orig.setPreferences(prefs)
    
    board = copy.deepcopy(board_orig)    
    
    if True:
    
        print("**** Test play")
        board.report()
        print(board.countriesPandas())
        
        for i in range(0):
          board.play()
          if board.gameOver: break
        
        print("\n\n**** End of play")
        board.report()
        print(board.countriesPandas())
    
    
    if False:
        print("\n\n")
        print("**** Test FlatMC, UCT and MCTS\n")
        
        # Get to the desired phase
        
        for _ in range(2):
            board.play()
        
        # while board.gamePhase != "attack":
        #     board.play()
        #     if board.gameOver: break

        board.report()
        print(board.countriesPandas())
        
        flat = FlatMC(max_depth = 300, sims_per_eval = 2, num_MCTS_sims = 2000, cb = 0)
        uct = UCT(max_depth = 300, sims_per_eval = 2, num_MCTS_sims = 2000, cb = 0.4)
        temp = 1
        p = board.activePlayer
        
        # bestAction, bestValue = flat.getBestAction(board, p.code, temp=1, num_sims = None, verbose=False)
        
        # print(f"Done flatMC: Player {p.code}")
        # actions = flat.As[hash(board)]
        # for i, a in enumerate(actions):
        #     if uct.Vs[hash(board)][i]:
        #         print(a, " -> ", flat.Rsa[(hash(board), hash(a))])
        
        # print()
        # print(bestAction)
        # print(bestValue)  
        # print(f"Length of the tree: {len(flat.Ns)}")
        
        
        bestAction, bestValue, R, Q = uct.getBestAction(board, p.code, num_sims = None, verbose=True)
        probs = uct.getVisitCount(board, temp=temp)
        print("----------------------------")
        print(f"Done UCT: Player {p.code}")
        actions = uct.As[hash(board)]
        for i, a in enumerate(actions):
            if uct.Vs[hash(board)][i]:
                print(a, " -> ", uct.Rsa[(hash(board), hash(a))])
        
        print()
        print("Best action: ", bestAction)
        print("Best value: ", bestValue)
        print("R: ", R)
        print("Q: ", Q)
        print("probs: ", probs)
        print(f"Length of the tree: {len(uct.Ns)}")
        
        
        
    
    #%%% Try PUCT    
    # Now try the network, and the MCTS with the network (apprentice and expert)
    if False:
        path_model = "../data/models"
        EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_small.json"
        
        # Create the net using the same parameters
        inputs = misc.read_json(EI_inputs_path)
        model_args =  misc.read_json(inputs["model_parameters"])
        board_params = inputs["board_params"]
        path_board = board_params["path_board"]

        # ---------------- Model -------------------------

        print("Creating board")
        
        
        world = World(path_board)


        # Set players
        pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Blue'), RandomAgent('Green')
        players = [pR1, pR2]
        # Set board
        prefs = board_params
                
        board_orig = Board(world, players)
        board_orig.setPreferences(prefs)

        num_nodes = board_orig.world.map_graph.number_of_nodes()
        num_edges = board_orig.world.map_graph.number_of_edges()

        print("Creating model")
        net = GCN_risk(num_nodes, num_edges, 
                         model_args['board_input_dim'], model_args['global_input_dim'],
                         model_args['hidden_global_dim'], model_args['num_global_layers'],
                         model_args['hidden_conv_dim'], model_args['num_conv_layers'],
                         model_args['hidden_pick_dim'], model_args['num_pick_layers'], model_args['out_pick_dim'],
                         model_args['hidden_place_dim'], model_args['num_place_layers'], model_args['out_place_dim'],
                         model_args['hidden_attack_dim'], model_args['num_attack_layers'], model_args['out_attack_dim'],
                         model_args['hidden_fortify_dim'], model_args['num_fortify_layers'], model_args['out_fortify_dim'],
                         model_args['hidden_value_dim'], model_args['num_value_layers'],
                         model_args['dropout'])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        
        load_model = False
        
        if load_model:
            # Choose a model at random
            model_name = np.random.choice(os.listdir(path_model))    
            print(f"Chosen model is {model_name}")
            state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')
            # print(state_dict)
            net.load_state_dict(state_dict['model'])
            print("Model has been loaded")
            
        
        num_sims = 100
        temp = 1
        num_plays = 0
        verbose = 1
        
        # Create player that uses neural net
        
        apprentice = NetApprentice(net)
        
        puct = PUCT(apprentice, max_depth = 200, sims_per_eval = 1, num_MCTS_sims = num_sims,
                 wa = 10, wb = 10, cb = 1.1, use_val = 0.5, console_debug = verbose)
        
        # Play some random moves, then use puct or player puct to tag the move (Expert move)
        
        board = copy.deepcopy(board_orig)
        
        # Test play
        for i in range(num_plays):
          board.play()
          if board.gameOver: break
  
        print("\n\n ***** End of play")  
        board.report()
        print(board.countriesPandas())
        
        
        print("\n\n ***** Playing PUCT")
        board.console_debug = False
        bestAction, bestValue, R, Q = puct.getBestAction(board, player = board.activePlayer.code, num_sims = num_sims, verbose=verbose)
        probs = puct.getVisitCount(board, temp=temp)
        
        
        print("\n\nExpert results")
        print("Action and value: ", bestAction, bestValue)
        print("R, Q: \n", R, "\n", Q)        
        print("probs: \n", probs)
        print("Actions (As): \n", puct.As[hash(board)])
        print("Policy (Ps): \n", puct.Ps[hash(board)])
        print("Valid (Vs): \n", puct.Vs[hash(board)])
        

    if False:
      # Test the net training. Get random boards, tag them with PUCT, then train
      
        ##### IMPORT FROM CREATE_SELF_PLAY
        import sys
        import time
        from model import saveBoardObs, save_dict
        def play_episode(root, max_depth, apprentice, move_type = "all", verbose=False):
            episode = []
            state = copy.deepcopy(root)
            edge_index = boardToData(root).edge_index
            # ******************* PLAY EPISODE ***************************
            for i in range(max_depth):  
                #print_message_over(f"Playing episode: {i}/{max_depth}")
        
                # Check if episode is over            
                if state.gameOver: break
        
                # Check is current player is alive or not
                if not state.activePlayer.is_alive: 
                    # print("\npassing, dead player")
                    state.endTurn()
                    continue
        
                # Get possible moves, and apprentice policy
                mask, actions = maskAndMoves(state, state.gamePhase, edge_index)
                
                try:
                    policy, value = apprentice.getPolicy(state)
                except Exception as e:
                    state.report()
                    print(state.activePlayer.is_alive)
                    print(state.activePlayer.num_countries)
                    raise e
                
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().numpy()
                
                probs = policy * mask             
                
                probs = probs.flatten()
                  
                probs =  probs / probs.sum()
        
                # Random selection? e-greedy?
                
                ind = np.random.choice(range(len(actions)), p = probs)
                move = buildMove(state, actions[ind])
                
                saved = (move_type=="all" or move_type==state.gamePhase)
                if verbose:             
                    # print(f"\t\tPlay episode: turn {i}, move = {move}, saved = {saved}")
                    pass
                
                if saved:
                    episode.append(copy.deepcopy(state))
                    
                if move_type == "initialPick":
                    if state.gamePhase != "initialPick":
                        break
                elif move_type == "initialFortify":
                    if state.gamePhase in ["startTurn", "attack", "fortify"]:
                        break
                        
                    
                # Play the move to continue
                state.playMove(move)
                
            return episode
             
        def create_self_play_data(move_type, path, root, apprentice, max_depth = 100, saved_states_per_episode=1, verbose = False):
            """ Function to create episodes from self play.
                Visited states are saved and then re visited with the expert to label the data        
            """
        
            if verbose: 
                print("\t\tSelf-play starting")
                sys.stdout.flush()
                
            try:
                # Define here how many states to select, and how
                # edge_index = boardToData(root).edge_index    
        
                # ******************* PLAY EPISODE ***************************
                episode = play_episode(root, max_depth, apprentice, move_type = move_type, verbose = verbose)
                
                # ******************* SELECT STATES ***************************
                # Take some states from episode    
                
                options = [s for s in episode if s.gamePhase == move_type]
                if not options:
                    # TODO: What to do in this case? For now just take some random states to avoid wasting the episode
                    options = episode
                states_to_save = np.random.choice(options, min(saved_states_per_episode, len(options)), replace=False)
            except Exception as e:
                raise e
            
            if verbose: 
                print(f"\t\tSelf-play done: move_type = {move_type}, {len(states_to_save)} states to save")
                sys.stdout.flush()
                
            return states_to_save
        
        
        def tag_with_expert_move(state, expert, temp=1, verbose=False):
            # Tag one state with the expert move    
            start = time.process_time()
            _, _, value_exp, Q_value_exp = expert.getBestAction(state, player = state.activePlayer.code, num_sims = None, verbose=verbose)
            policy_exp = expert.getVisitCount(state, temp=temp)
            # TODO: Katago improvement
            
            if isinstance(policy_exp, torch.Tensor):
                policy_exp = policy_exp.detach().numpy()
            if isinstance(value_exp, torch.Tensor):
                value_exp = value_exp.detach().numpy()
            
            if verbose: 
                print(f"\t\tTag with expert: Tagged board {state.board_id} ({state.gamePhase}). {round(time.process_time() - start,2)} sec")
                sys.stdout.flush()
            
            return state, policy_exp, value_exp
            
        
        def simple_save_state(root_path, state, policy, value, verbose=False, num_task=0):
            try:
                board, _ = state.toCanonical(state.activePlayer.code)
                phase = board.gamePhase
                full_path = os.path.join(root_path, phase, 'raw')
                num = len(os.listdir(full_path))+1        
                name = f"board_{num}.json"
                while os.path.exists(os.path.join(full_path, name)):
                    num += 1
                    name = f"board_{num}.json"
                name = f"board_{num}_{num_task}.json" # Always different
                saveBoardObs(full_path, name,
                                    board, board.gamePhase, policy.ravel().tolist(), value.ravel().tolist())
                if verbose: 
                    print(f"\t\tSimple save: Saved board {state.board_id} {os.path.join(full_path, name)}")
                    sys.stdout.flush()
                return True
            except Exception as e:
                print(e)
                raise e
                
                
        ##### end IMPORT FROM CREATE_SELF_PLAY ############
        
        
        # Define inputs here
        load_model = True
        max_depth = 150
        move_type = "startTurn"
        path_data = "../data_diamond_test"
        path_model = "../data_diamond_test/models"
        num_sims = 600
        saved_states_per_episode=1
        verbose = 1
        num_states = 4
        batch_size = 4
        temp = 1
        
        EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_small.json"
        
        
        # Create folders if they dont exist
        move_types = ["initialPick", "initialFortify", "startTurn", "attack", "fortify"]
        for folder in move_types:
            os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
        os.makedirs(path_model, exist_ok = True)
        
        # Create the net using the same parameters
        inputs = misc.read_json(EI_inputs_path)
        model_args =  misc.read_json(inputs["model_parameters"])
        board_params = inputs["board_params"]
        path_board = board_params["path_board"]

        # ---------------- Model -------------------------

        print("Creating board")
        
        world = World(path_board)


        # Set players
        pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Blue'), RandomAgent('Green')
        players = [pR1, pR2]
        # Set board
        prefs = board_params
                
        board_orig = Board(world, players)
        board_orig.setPreferences(prefs)

        num_nodes = board_orig.world.map_graph.number_of_nodes()
        num_edges = board_orig.world.map_graph.number_of_edges()

        ##### Create the net 
        
        print("Creating model")
        net = GCN_risk(num_nodes, num_edges, 
                         model_args['board_input_dim'], model_args['global_input_dim'],
                         model_args['hidden_global_dim'], model_args['num_global_layers'],
                         model_args['hidden_conv_dim'], model_args['num_conv_layers'],
                         model_args['hidden_pick_dim'], model_args['num_pick_layers'], model_args['out_pick_dim'],
                         model_args['hidden_place_dim'], model_args['num_place_layers'], model_args['out_place_dim'],
                         model_args['hidden_attack_dim'], model_args['num_attack_layers'], model_args['out_attack_dim'],
                         model_args['hidden_fortify_dim'], model_args['num_fortify_layers'], model_args['out_fortify_dim'],
                         model_args['hidden_value_dim'], model_args['num_value_layers'],
                         model_args['dropout'])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        
        
        ##### Load the model to use as apprentice
        
        if load_model:
            # Choose a model at random
            model_name = np.random.choice(os.listdir(path_model))    
            print(f"Chosen model is {model_name}")
            state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')
            # print(state_dict)
            net.load_state_dict(state_dict['model'])
            print("Model has been loaded")
            
        
        ###### Get some random games and tag them
        # Define apprentice and expert 
        apprentice = NetApprentice(net)        
        expert = PUCT(apprentice, max_depth = max_depth, sims_per_eval = 1, num_MCTS_sims = num_sims,
                 wa = 10, wb = 10, cb = 1.1, use_val = 0.5, console_debug = verbose)
        
        # Play
        state = copy.deepcopy(board_orig)
        states_to_save = []
        for _ in range(num_states):
            aux = create_self_play_data(move_type, path_data, state, apprentice, max_depth = max_depth, saved_states_per_episode=saved_states_per_episode, verbose = verbose)
            states_to_save.extend(aux)
        
        
        # Tag and save
        for k, st in enumerate(states_to_save):
            print(f"Tagging state {k}")
            st_tagged, policy_exp, value_exp = tag_with_expert_move(st, expert, temp=temp, verbose=verbose)
            res = simple_save_state(path_data, st_tagged, policy_exp, value_exp, verbose=verbose, num_task=k)
        
        
        
        # Now train the model
        
        # Create the dataset
        from model import RiskDataset
        from torch_geometric.data import DataLoader as G_DataLoader
        
        save_path = f"{path_model}/model_test_test_{move_type}.tar"
        root_path = f'{path_data}/{move_type}'
        
        if len(os.listdir(os.path.join(root_path, 'raw')))<batch_size:
            raise Exception("Not enough data")
        
        risk_dataset = RiskDataset(root = root_path)        
        
        train_loader = G_DataLoader(risk_dataset, batch_size=batch_size, shuffle = True)

        # Create optimizer and scheduler
        
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        from model import total_Loss
        criterion = total_Loss
        
       
        for batch, global_batch in train_loader:   
            batch.to(device)
            global_batch.to(device) 
            optimizer.zero_grad()
            try:
                pick, place, attack, fortify, value = net.forward(batch, global_batch)
            except Exception as e:
                print("Batch: \n", batch.x)
                print(batch.x.shape)
                print("global: ", global_batch.shape)
                raise e
            # print("pick", pick)
            # print("place", place)
            # print("attack", attack)
            # print("fortify", fortify)
            # print("value", value)
            phase = batch.phase[0]
            if phase == 'initialPick':
                out = pick
            elif phase in ['initialFortify', 'startTurn']:
                out = place
            elif phase == 'attack':
                out = attack
            elif phase == 'fortify':
                out = fortify            
            
            y = batch.y.view(batch.num_graphs,-1)
            z = batch.value.view(batch.num_graphs,-1)
            # print("out\n", out)
            # print("y\n", y)
            loss = criterion(out, value, y, z)
            # print("out: ", out)
            # print("value: ", value)
            # print("target: ", y)
            # print("value target: ", z)
            # print(loss)
            loss.backward()
            optimizer.step()
            
        save_dict(save_path, {'model':net.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'epoch': 0, 'best_loss':0})
        
    if False:
      
        # Test NetPlayer
        path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_test_map/models"
        EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs.json"
        
        # Create the net using the same parameters
        inputs = misc.read_json(EI_inputs_path)
        model_args =  misc.read_json(inputs["model_parameters"])
        board_params = inputs["board_params"]
        path_board = board_params["path_board"]

        # ---------------- Model -------------------------

        print("Creating board")
        
        
        world = World(path_board)


        # Set players
        pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Blue'), RandomAgent('Green')
        players = [pR1, pR2]
        # Set board
        prefs = board_params
                
        board_orig = Board(world, players)
        board_orig.setPreferences(prefs)



        num_nodes = board_orig.world.map_graph.number_of_nodes()
        num_edges = board_orig.world.map_graph.number_of_edges()

        print("Creating model")
        net = GCN_risk(num_nodes, num_edges, 
                         model_args['board_input_dim'], model_args['global_input_dim'],
                         model_args['hidden_global_dim'], model_args['num_global_layers'],
                         model_args['hidden_conv_dim'], model_args['num_conv_layers'],
                         model_args['hidden_pick_dim'], model_args['num_pick_layers'], model_args['out_pick_dim'],
                         model_args['hidden_place_dim'], model_args['num_place_layers'], model_args['out_place_dim'],
                         model_args['hidden_attack_dim'], model_args['num_attack_layers'], model_args['out_attack_dim'],
                         model_args['hidden_fortify_dim'], model_args['num_fortify_layers'], model_args['out_fortify_dim'],
                         model_args['hidden_value_dim'], model_args['num_value_layers'],
                         model_args['dropout'])

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device)
        
        load_model = True
        model_name = "model_9_4_initialPick.tar"
        
        if load_model:
            if model_name is None:
                # Choose a model at random
                model_name = np.random.choice(os.listdir(path_model))    
            print(f"Chosen model is {model_name}")
            state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')
            # print(state_dict)
            net.load_state_dict(state_dict['model'])
            print("Model has been loaded\n\n")
        
                
                    
        # Create player that uses neural net
        
        apprentice = NetApprentice(net)
        netPlayer = NetPlayer(apprentice, move_selection = "random_proportional")
        
        players = [netPlayer, pR2]
        # Set board
        prefs = board_params
        board = Board(world, players)
        board.setPreferences(prefs)
        board.console_debug=True
                
        
        # Test play
        for i in range(50):
          board.play()
          if board.gameOver: break
  
        print("\n\n ***** End of play")  
        board.report()
        print(board.countriesPandas())
 
