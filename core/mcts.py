from move import Move

class MCTS(object):
  ''' Class to perform Monte Carlo Tree Search '''
  def __init__(self, rootPlayer, leafFunc):
    self.Qsa, self.Nsa, self.Ns, self.Ps = {}, {}, {}, {}
    self.rootPlayer = rootPlayer
    self.leafFunc = leafFunc

  def expandCondition(self, state, args):
    ''' Determines when to expand a leaf.
        Example: for flatMC or UCB, only the root is expanded (args['depth']==0)
        For UCT or any other common MCTS algorithm, the leaf is always expanded
    '''
    return True
    
  def search(self, state, args):
    ''' Performs the search in the known tree following the tree policy.
    Once a leaf is found, the expand and evaluation functions are called to
    develop the tree and get an evaluation for the new node
    '''
    currPlayer = state.activePlayer
    # 1. If state is terminal, return score
    v = Move.isTerminal(state, self.rootPlayer)
    if v != 0: return v

    # 2. If is leaf
    if not hash(state) in self.Ps:
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
      if self.expandCondition(state, args): self.Ps[hash(state)] = pi
      if hash(state) in self.Ns:
        self.Ns[hash(state)] += 1
      else:
        self.Ns[hash(state)] = 0
      return v

    # 3. Not a leaf, continue search
    
      
    
    
    
