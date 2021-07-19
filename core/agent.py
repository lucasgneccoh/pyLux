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
import json
import os
import sys

from board import Board, Agent, RandomAgent
from world import World

import torch

# Needs torch_geometric
from model import boardToData, buildGlobalFeature, GCN_risk, load_dict, save_dict
import torch_geometric




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
            policy, value = np.zeros(len(moves)), np.zeros(6) 
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

        
        # Expand 
        batch = torch_geometric.data.Batch.from_data_list([boardToData(state)])
        mask, moves = maskAndMoves(state, state.gamePhase, batch.edge_index)
                
        policy, value = np.zeros(len(moves)), np.zeros(6) 
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
                    sc = uct
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
                  cb = np.sqrt(2)):        
        self.name = name
        self.human = False
        self.console_debug = False 
        self.max_depth = max_depth
        self.sims_per_eval = sims_per_eval
        self.num_MCTS_sims = num_MCTS_sims
        self.cb = cb
        self.uct = UCT(max_depth, sims_per_eval, num_MCTS_sims, cb)
        
 
    def run(self, board, num_sims=None):   
        self.uct = UCT(self.max_depth, self.sims_per_eval, self.num_MCTS_sims, self.cb)    
        bestAction, bestValue, _, _ = self.uct.getBestAction(board, self.code, num_sims = num_sims, verbose=False)
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
            policy, value = self.apprentice.getPolicy(canon)
        else:
            # No bias
            policy, value = torch.ones_like(mask)/max(mask.shape), torch.zeros((1,6))
                    
        
        #if self.console_debug: print(f"OnLeaf: State {state.board_id} ({s})")
        #if self.console_debug: print(f"OnLeaf: Found this actions to expand {moves}")
        
        if not isinstance(policy, torch.Tensor):
            policy = torch.Tensor(policy)
            
        policy = policy * mask
        self.Vs[s], self.As[s] = mask.squeeze().detach().numpy(), moves
        self.Ps[s] = policy.squeeze().detach().numpy()
        self.Ns[s] = 1

        if not self.use_val:
            # Return an evaluation
            v = np.zeros(6)
            for _ in range(self.sims_per_eval):
                sim = copy.deepcopy(state)
                sim.simulate()
                v += score_players(sim)                
            v /= self.sims_per_eval
            cor_value = v
        else:
            # Fix order of value returned by net
            value = value.squeeze()
            # Apprentice already does this
            cor_value = np.array([value[map_to_orig.get(i)] if not map_to_orig.get(i) is None else 0.0  for i in range(6)])
        
        # Value estimated by the network, value estimated by the network
        return cor_value, cor_value
        
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
                    """
                    uct = self.Rsa[(s,a)][p]+ self.cb * np.sqrt(np.log(self.Ns[s]) / max(self.Nsa[(s,a)], self.eps))
                    pol = 0 if self.apprentice is None else self.wa * self.Ps[s][i]/(self.Nsa[(s,a)]+1)
                    val = [0]*6 if self.apprentice is None else self.wb * self.Qsa[(s,a)] * (self.use_val)
                    sc = uct + pol + val[p]
                    """
                    mean = self.Rsa[(s,a)][p]
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
            rsa, nsa = self.Rsa[(s,a)], self.Nsa[(s,a)]
            self.Rsa[(s,a)] = (nsa*rsa + v) /(nsa + 1)
            # self.Qsa[(s,a)] = (nsa*qsa + net_v) /(nsa + 1)
            self.Nsa[(s,a)] += 1
        
        else:
            self.Rsa[(s,a)] = v
            # self.Qsa[(s,a)] = net_v
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


#%% Neural MCTS
    


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
      z = np.random.uniform()
      if self.move_selection == "argmax":
          ind = np.argmax(probs)
      elif self.move_selection == "random_proportional":
          ind = np.random.choice(range(len(actions)), p = probs)
      else:
          raise Exception("Invalid kind of move selection criterion")
      
      # Return the selected move, destroy tree
      self.puct = None
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
    

def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--inputs", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/exp_iter_inputs.json")
  args = parser.parse_args()
  return args
  


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
    path = '../support/maps/mini_test_map.json'
      
    world = World(path)
    

    # Set players  
    pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Green'), RandomAgent('Blue')
    
    pFlat = FlatMCPlayer(name='flatMC', max_depth = 300, sims_per_eval = 2, num_MCTS_sims = 700, cb = 0)
    pUCT = UCTPlayer(name='UCT', max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000, cb = np.sqrt(2))
    
    players = [pR1, pFlat]
    # Set board
    prefs = {'initialPhase': True, 'useCards':True,
             'transferCards':True, 'immediateCash': True,
             'continentIncrease': 0.05, 'pickInitialCountries':True,
             'armiesPerTurnInitial':4,
             'console_debug':True}  
             
    board_orig = Board(world, players)
    board_orig.setPreferences(prefs)
    
    board = copy.deepcopy(board_orig)    
    
    if False:
    
        print("**** Test play")
        board.report()
        print(board.countriesPandas())
        
        for i in range(50):
          board.play()
          if board.gameOver: break
        
        print("\n\n**** End of play")
        board.report()
        print(board.countriesPandas())
    
    
    if False:
        print("\n\n")
        print("**** Test FlatMC, UCT and MCTS\n")
        
        # Get to the desired phase
        while board.gamePhase != "attack":
            board.play()
            if board.gameOver: break

        board.report()
        print(board.countriesPandas())
        flat = FlatMC(max_depth = 300, sims_per_eval = 2, num_MCTS_sims = 2000, cb = 0)
        uct = UCT(max_depth = 300, sims_per_eval = 2, num_MCTS_sims = 2000, cb = 0.4)
        
        p = board.activePlayer
        
        bestAction, bestValue = flat.getBestAction(p.code, board, temp=1, num_sims = None, verbose=False)
        
        print(f"Done flatMC: Player {p.code}")
        actions = flat.As[hash(board)]
        for i, a in enumerate(actions):
            print(a, " -> ", flat.Rsa[(hash(board), hash(a))])
        
        print()
        print(bestAction)
        print(bestValue)  
        print(f"Length of the tree: {len(flat.Ns)}")
        
        
        bestAction, bestValue = uct.getBestAction(p.code, board, temp=1, num_sims = None, verbose=False)
        print("----------------------------")
        print(f"Done UCT: Player {p.code}")
        actions = uct.As[hash(board)]
        for i, a in enumerate(actions):
            print(a, " -> ", uct.Rsa[(hash(board), hash(a))])
        
        print()
        print(bestAction)
        print(bestValue)   
        print(f"Length of the tree: {len(uct.Ns)}")
        
        
        
    
        
    # Now try the network, and the MCTS with the network (apprentice and expert)
    if True:
        path_model = "../data/models"
        EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs.json"
        
        # Create the net using the same parameters
        inputs = misc.read_json(EI_inputs_path)
        model_args =  misc.read_json(inputs["model_parameters"])
        board_params = inputs["board_params"]
        path_board = board_params["path_board"]

        # ---------------- Model -------------------------

        print("Creating board")

        #%%% Create Board
        
        world = World(path_board)


        # Set players
        pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Blue'), RandomAgent('Green')
        players = [pR1, pR2]
        # Set board
        # TODO: Send to inputs
        prefs = {'initialPhase': True, 'useCards':True,
                'transferCards':True, 'immediateCash': True,
                'continentIncrease': 0.05, 'pickInitialCountries':True,
                'armiesPerTurnInitial':4,'console_debug':True}
                
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
        
        if load_model:
            # Choose a model at random
            model_name = np.random.choice(os.listdir(path_model))    
            print(f"Chosen model is {model_name}")
            state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')
            print(state_dict)
            net.load_state_dict(state_dict['model'])
            print("Model has been loaded")
            
        
        
        print("\nReceived args:\n")
        print(sys.argv)        
        
        num_sims = int(sys.argv[1])
        temp = int(sys.argv[2])
        num_plays = int(sys.argv[3])
        verbose = int(sys.argv[4]) if len(sys.argv)>4 else 0
        
        # Create player that uses neural net
        
        apprentice = NetApprentice(net)
        
        puct = PUCT(apprentice, max_depth = 200, sims_per_eval = 1, num_MCTS_sims = 1000,
                 wa = 10, wb = 10, cb = 0.4, use_val = 0, console_debug = verbose)
        
        # Play some random moves, then use puct or player puct to tag the move (Expert move)
        
        board = copy.deepcopy(board_orig)
        
        
        
        # Test play
        for i in range(num_plays):
          board.play()
          if board.gameOver: break
  
        print("\n\n *** End of play")  
        board.report()
        print(board.countriesPandas())
        
        
        print("\n\n Playing PUCT")
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
        

 
