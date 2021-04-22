# -*- coding: utf-8 -*-
"""!@package pyRisk

The agent module contains the pyRisk players.
"""

import numpy as np
import pyRisk

class Agent(object):
  '''!Base class for an agent
  
  All players should extend from this class
  '''  
  def __init__(self, name='agent'):
  '''! Constructor. By default sets the *console_debug* and *human* attributes to **False**
  
  :param name: The name of the agent
  :type name: str
  '''
    self.name_string = name
    self.human = False
    self.console_debug = False
  
  def setPrefs(self, code:int, board):
  '''! Puts the agent into the game by assigning it a code and giving it the reference to the game board
  
  :param code: The internal code of the agent in the game.
  :type code: int
  :param board: Reference to the board with the game information
  :type :py:module:`pyRisk`.:py:class:`Board`
  '''
    self.code = code
    self.board = board
    
  
  def pickCountry(self):
  '''! Choose an empty country at the beginning of the game
  '''
    pass
    
  def placeInitialArmies(self,numberOfArmies:int):
  '''! Place armies in owned countries. No attack phase follows.
  Use self.board.placeArmies(numberArmies:int, country:Country)
  
  :param numberOfArmies: The number of armies the agent must place on the board
  :type numberOfArmies: int
  '''
    pass
  
  
  def cardPhase(self, cards):
  '''! Call to exchange cards.
  May return None, meaning no cash
  Use :py:module:`pyRisk`.:py:module:`Deck` methods to check for sets and get best sets
  
  :param cards: List with the player's cards
  :type cards: list[:py:module:`pyRisk`.:py:class:`Card`]
  :returns List of three cards to cash
  :rtype list[:py:module:`pyRisk`.:py:class:`Card`]
  '''
    pass
  
  
  def placeArmies(self, numberOfArmies:int):
  '''! Given a number of armies, place them on the board
  Use self.board.placeArmies(numberArmies:int, country:Country) to place the armies
  
  :param numberOfArmies: The number of armies the agent must place on the board
  :type numberOfArmies: int
  '''
    pass
  
  
  def attackPhase(self):
  '''! Call to attack. 
  Can attack till dead or make single rolls.
  Use res = self.board.attack(source_code:int, target_code:int, tillDead:bool) to do the attack
  Every call to board.attack yields a result as follows:
    - 7: Attacker conquered the target country. This will call to moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int) to determine the number of armies to move to the newly conquered territory
    - 13: Defender won. You are left with only one army and can therefore not continue the attack
    - 0: Neither the attacker nor the defender won. If you want to deduce the armies you lost, you can demand the number of armies in your country.
    - -1: There was an error
    
  You can attack multiple times during the attack phase.
  '''
    pass
    
  def moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int) -> int:
  '''! This method is called when an attack led to a conquer. 
  You can choose the number of armies to send to the conquered territory. 
  The returned value should be between the number of rolled dice and the number of armies in the attacking country minus 1.
  The number of dice rolled is always 3 except when you have 3 or less armies, in which case it will be the number of attacking armies minus 1.
  Notice that by default, this method returns all the movable armies.
  
  
  :param countryCodeAttacker: Internal code of the attacking country
  :type countryCodeAttacker: int
  :param countryCodeDefender: Internal code of the defending country
  :type countryCodeDefender: int
  :returns Number of armies to move from attacking country to newly conquered territory.
  :rtype int
  '''
    # Default is to move all but one army
    return self.board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self):
  '''! Call to fortify.
  Just before this call, board will update the movable armies in each territory to the number of armies.
  You can fortify multiple times, always between your countries that are linked and have movable armies.
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

class RandomAgent(Agent):

  def __init__(self, name='random', aggressiveness = 0.5):
    '''! Constructor of random agent.
    
    :param name: Name of the agent.
    :type name: str
    :param aggressiveness: Level of aggressiveness. Determines the probability of attacking until dead. 1 means always attacking until dead when attacking.
    :type aggressiveness: float
    '''
    super().__init__(name)
    self.aggressiveness = aggressiveness
    
  
  def pickCountry(self):
    '''! Pick at random one of the empty countries
    '''    
    options = self.board.countriesLeft
    if options:
      return np.random.choice(options)
    else:
      return None
  
  def placeInitialArmies(self, numberOfArmies:int):
    '''! Pick at random one of the empty countries
    '''  
    countries = self.board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      if self.console_debug: print(f"{self.name()}: Placing 1 armies in {c.id}")
      self.board.placeArmies(1, c)
  
  def cardPhase(self, cards):
    '''! Only cash when forced, then cash best possible set
    '''
    if len(cards)<5 or not pyRisk.Deck.containsSet(cards): 
      return None
    c = pyRisk.Deck.yieldBestCashableSet(cards, self.code, self.board.countries)
    if not c is None:      
      return c
  
  def placeArmies(self, numberOfArmies:int):
    '''! Place armies at random one by one
    '''
    countries = self.board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      self.board.placeArmies(1, c)
  
  def attackPhase(self):
    '''! Attack a random number of times, from random countries, to random targets.
    The till_dead parameter is also set at random using an aggressiveness parameter
    '''  
    canAttack = [c for c in self.board.getCountriesPlayer(self.code) if c.armies > 1 and c.getNumberEnemyNeighbors(kind=1)>0]
    if len(canAttack)==0: return
    nbAttacks = np.random.randint(len(canAttack))
    for _ in range(nbAttacks):
      source = np.random.choice(canAttack)
      target = np.random.choice(source.getHostileAdjoiningCodeList())
      target_c = self.board.world.countries[target]
      tillDead = True if np.random.uniform()<self.aggressiveness else False
      res = self.board.attack(source.code, target_c.code, tillDead)
      if self.console_debug: print(f"{self.name()}: Attacking {source.id} -> {target_c.id}: tillDead {tillDead}: Result {res}")
    
    return 
  
  def fortifyPhase(self):
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
  
  
all_agents = {'random': RandomAgent, 'human': Human}
    