# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:36:56 2021

@author: lucas
"""
#%% Agents
import numpy as np
import pyRisk

class Agent(object):
  '''
  Base class for an agent
  '''  
  def __init__(self, name='agent'):
    self.name_string = name
    self.human = False
    self.console_debug = False
  
  def setPrefs(self, code:int, board):
    self.code = code
    self.board = board
    
  
  def pickCountry(self):
    pass
  
  def placeInitialArmies(self,numberOfArmies:int):
    pass
  
  def cardPhase(self, cards):
    pass
  
  def placeArmies(self, numberOfArmies:int):
    pass
  
  def attackPhase(self):
    pass
  
  def moveArmiesIn(self, countryCodeAttacker:int, countryCodeDefender:int):
    # Default is to move all but one army
    return self.board.world.countries[countryCodeAttacker].armies-1

  def fortifyPhase(self):
    pass
  
  def name(self):
    return self.name_string
  
  def version(self):
    return '0'
  
  def description(self):
    return 'Base description of agent'  
  
  def youWon(self):
    return '{} won the game'.format(self.name_string)
  
  def message(self):
    return 'Hello human, this is {}'.format(self.name_string)  

class RandomAgent(Agent):

  def __init__(self, name='random'):
    super().__init__(name)
    
  
  def pickCountry(self):
    '''
    To make it faster, create a list of remaining countries in board
    '''
    # options = [c.code for c in board.getCountries() if c.owner == -1]
    options = self.board.countriesLeft
    if options:
      return np.random.choice(options)
    else:
      return None
  
  def placeInitialArmies(self, numberOfArmies:int):
    countries = self.board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      if self.console_debug: print(f"{self.name()}: Placing 1 armies in {c.id}")
      self.board.placeArmies(1, c)
  
  def cardPhase(self, cards):
    if len(cards)<5 or not pyRisk.Deck.containsSet(cards): 
      return 0
    c = pyRisk.Deck.yieldBestCashableSet(cards, self.code, self.board.countries)
    if not c is None:
      armies = board.cashCards(*c)
      print(armies)
      if self.console_debug: print(f"Cashed cards: {armies}")
      return armies
  
  def placeArmies(self, numberOfArmies:int):
    countries = self.board.getCountriesPlayer(self.code)
    for _ in range(numberOfArmies):
      c = np.random.choice(countries)
      self.board.placeArmies(1, c)
  
  def attackPhase(self):
      
    canAttack = [c for c in self.board.getCountriesPlayer(self.code) if c.armies > 1 and c.getNumberEnemyNeighbors(kind=1)>0]
    if len(canAttack)==0: return
    source = np.random.choice(canAttack)
    target = np.random.choice(source.getHostileAdjoiningCodeList())
    target_c = self.board.world.countries[target]
    tillDead = True if np.random.uniform()>0.5 else False
    
    # For now, attack one time and pass to next step
    res = self.board.attack(source.code, target_c.code, tillDead)
    
    if self.console_debug: print(f"{self.name()}: Attacking {source.id} -> {target_c.id}: tillDead {tillDead}: Result {res}")
    
    return 
  

  def fortifyPhase(self):
    # For now, no fortification is made
    if self.console_debug: print(f"{self.name()}: Fortify: Nothing")
    return 
  
class Human(Agent):
  def __init__(self, name='human'):
    super().__init__(name)
    self.human = True
    
  def message(self):
    return 'I am human'
  
  def version(self):
    return 'Homo sapiens sapiens'
  
  
all_agents = {'random': RandomAgent, 'human': Human}
    