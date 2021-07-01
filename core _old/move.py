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
  
    
