import itertools
class Move(object):
  '''! Class used to simpify the idea of legal moves for an agent
  A move is just a tuple (source, target, armies) where wource and target
  are countries, and armies is an integer.
  In the initial pick, initial fortify and start turn phases, source = target
  is just the chosen country.
  In the attack and fortify phases, the number of legal moves can be very big
  if every possible number of armies is considered, so we may limit to some
  options (1, 5, all) 
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

  @staticmethod
  def isTerminal(board, currPlayer):
    if board.getNumberOfPlayersLeft()==1:
      if board.players[currPlayer].is_alive: return 1
      return -1
    # Not over
    return 0
    
