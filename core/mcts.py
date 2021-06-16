from board import Board
from world import World, Country, Continent
from move import Move
import numpy as np
import copy
import torch
from model import boardToData, buildGlobalFeature

import torch_geometric

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
        self.apprentice = MCTS(root = None, apprentice = None,
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

    def __init__(self, root, apprentice, max_depth = 50, sims_per_eval = 1, num_MCTS_sims = 1000,
                 wa = 10, wb = 10, cb = np.sqrt(2)):
        """ apprentice = None is regular MCTS
            apprentice = neural net -> Expert iteration with policy (and value) net
            apprentice = MCTS -> Used for first iteration
        """
        self.root, self.apprentice = root, apprentice
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
                policy, value = torch.ones_like(mask)/max(mask.shape), torch.zeros((1,6))
                        
            policy = policy * mask
            self.Vs[s], self.As[s] = mask.squeeze(), moves
            self.Ps[s] = policy.squeeze()
            self.Ns[s] = 1

            # Return an evaluation
            v = np.zeros(6)
            for _ in range(self.sims_per_eval):
                sim = copy.deepcopy(state)
                sim.simulate(agent.RandomAgent())
                v += score_players(sim)                
            v /= self.sims_per_eval

            # Fix order of value returned by net
            value = value.squeeze()
            cor_value = torch.FloatTensor([value[map_to_orig.get(i)] if not map_to_orig.get(i) is None else 0.0  for i in range(6)])
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

    def start_search(self, use_val = False, show_final=False):
        state = copy.deepcopy(self.root)
        v, net_v = self.search(state, 0, use_val)
        if show_final:
            print("++++++++++ FINAL ++++++++++")
            state.report()
        return v, net_v

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

if __name__ == '__main__':
  state = copy.deepcopy(board)
  state.report()
  mcts = None
  mcts = MCTS(root=state, apprentice=None, 
              max_depth=50, sims_per_eval=5, num_MCTS_sims=100)
  probs, R, Q = mcts.getActionProb(mcts.root, temp=1)
  print(probs)
  print(R)
  print(Q)
  m = np.argmax(probs).item()
  move = buildMove(mcts.root, mcts.As[hash(mcts.root)][m])
  print(move)