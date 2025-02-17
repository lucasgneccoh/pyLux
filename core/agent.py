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



class HumanAgent(Agent):

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
        return Move(s, s, move[2], phase)


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





def removekey(d, key):
    r = dict(d)
    del r[key]
    return r
    
def create_player_list(args):
    # Only need board_params and players in args
    board_params = args["board_params"]    

    list_players = []
    for i, player_args in enumerate(args["players"]):
        kwargs = removekey(player_args, "agent")
        if player_args["agent"] == "RandomAgent":
            list_players.append(RandomAgent(f"Random_{i}"))
        elif player_args["agent"] == "PeacefulAgent":
            list_players.append(PeacefulAgent(f"Peaceful_{i}"))
        elif player_args["agent"] == "FlatMCPlayer":
            list_players.append(FlatMCPlayer(name=f'flatMC_{i}', **kwargs))
        elif player_args["agent"] == "UCTPlayer":
            list_players.append(UCTPlayer(name=f'UCT_{i}', **kwargs))
        elif player_args["agent"] == "Human":
            hp_name = player_args["name"] if "name" in player_args else "human"
            hp = HumanAgent(name=hp_name)
            list_players.append(hp)
            
    return list_players



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
    
    if False:
    
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
        
        
        
    
        
