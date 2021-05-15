from move import Move
import numpy as np
import agent

class MCTS(object):
  ''' Class to perform Monte Carlo Tree Search '''
  def __init__(self, rootPlayer):
    self.Qsa, self.Nsa, self.Ns, self.Ps = {}, {}, {}, {}
    self.rootPlayer = rootPlayer
    self.eps = 1e-8

  def expandCondition(self, state, args):
    ''' Determines when to expand a leaf.
        Example: for flatMC or UCB, only the root is expanded (args['depth']==0)
        For UCT or any other common MCTS algorithm, the leaf is always expanded
    '''
    return True

  def leafFunc(self, state, args):
    ''' When applied to a leaf, gives a distribution over actions from the
        current state, and the value of the leaf
        By default, we expand and perform a rollout
    '''
    valid = Move.buildLegalMoves(state)
    copy_state = copy.deepcopy(state)
    copy_state.simulate(agent.RandomAgent, 0, changeAllAgents=True)
    return np.ones((1, len(valid)))/len(valid), Move.isTerminal(copy_state, args.rootPlayer)

  def selectAction(self, legalMoves, state, args):
    ''' Given a list of possible actions, select the action to perform
        Default is UCB1
    '''
    c = np.sqrt(2)
    s, a = hash(s), hash(legalMoves[0])
    bMove, bScore = legalMoves[0], self.Qsa[(s,a)] + \
                    c*np.sqrt(np.log(self.Ns[s]) / max(self.Nsa[(s,a)], self.eps))
    for i, move in legalMoves[1:]:
      a = hash(move)
      score = self.Qsa[(s,a)] + \
            c*np.sqrt(np.log(self.Ns[s]) / max(self.Nsa[(s,a)], self.eps))
      if score > bScore:
        bMove = move
        bScore = score
    return bMove

  def playAction(self, state, action):
    ''' Advances the game from state by performing action
        Modifies the state inplace
    '''
    Move.play(state, action)
    return
    
    
  def search(self, state, args):
    ''' Performs the search in the known tree following the tree policy.
    Once a leaf is found, the expand and evaluation functions are called to
    develop the tree and get an evaluation for the new node
    '''
    currPlayer = state.activePlayer
    s = hash(state)
    
    # 1. If state is terminal, return score
    v = Move.isTerminal(state, self.rootPlayer)
    if v != 0: return v

    # 2. If is leaf
    if not s in self.Ps:
      # 2.1 Evaluate
      # TODO: hay que hacer que tengan la misma longitud/estructura
      # Esto entonces depende de la fase del juego
      # TODO: Como hacer que sea general? para flat MC, UCB por ejemplo
      pi, v = self.leafFunc(state) # pi must have size of legalMoves

      # If pi is None, that means no expansion must be made
      # i.e FlatMC
      # 2.2 Get valid moves
      legalMoves = Move.buildLegalMoves(state)

      # 2.3 Mask policy and save it
      pi = legalMoves * pi # Put 0 on invalid moves
      if self.expandCondition(state, args): self.Ps[s] = pi
      if s in self.Ns:
        self.Ns[s] += 1
      else:
        self.Ns[s] = 0
      return v

    # 3. Not a leaf, continue search
    legalMoves = Move.buildLegalMoves(state)
    action = self.selectAction(legalMoves, state, args)
    a = hash(action)
    self.playAction(state, action)

    args['depth'] += 1
    v = self.search(state, args)

    # 4. Backpropagate
    if (s,a) in self.Qsa:
      self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)] + 1)
      self.Nsa[(s,a)] += 1
    else:
      self.Qsa[(s,a)] = v
      self.Nsa[(s,a)] = 1

    self.Ns[s] += 1
    return v
      
    
    
    
