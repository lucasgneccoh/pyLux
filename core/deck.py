# -*- coding: utf-8 -*-

#%% Card sequence

import random

class AbstractCardSequence(object):
  '''! Represents a sequence of numbers corresponding to the armies cashed in the game
  '''
  def __init__(self):
    self.cont = 0
  
  def nextCashArmies(self):
    '''! Default sequence is just 1, 2, 3, 4...'''
    self.cont += 1
    return self.cont


class GeometricCardSequence(AbstractCardSequence):
  '''! Geometric sequence of the form 
  s_n = base*(1 + incr)^n
  '''
  def __init__(self, base = 6, incr = 0.03):
    super().__init__()
    self.geometric = base    
    self.factor = (1.0+incr)
  
  def nextCashArmies(self):
    armies = int(self.geometric)
    self.geometric *= self.factor
    return armies

class ArithmeticCardSequence(AbstractCardSequence):
  '''! Arithmetic sequence of the form 
  s_n = base + incr*n
  '''
  def __init__(self, base = 4, incr = 2):
    super().__init__()
    self.arithmetic = base
    self.incr = incr    
  
  def nextCashArmies(self):
    armies = int(self.arithmetic)
    self.arithmetic += self.incr
    return armies

class ListCardSequence(AbstractCardSequence):
  '''! Sequence given by a fixed list. Once the list is exhausted, the sequence becomes constant yielding always the last element of the list
  '''
  def __init__(self, sequence):
    super().__init__()
    self.sequence = sequence
    self.M = len(sequence)-1
  
  def nextCashArmies(self):
    armies = self.sequence[min(self.cont, self.M)]
    self.cont += 1
    return armies

class ListThenArithmeticCardSequence(AbstractCardSequence):
  '''! Sequence given by a fixed list but then continued with an arithmetic sequence once the list is exhausted.
  '''
  def __init__(self, sequence, incr):
    super().__init__()
    self.sequence = sequence
    self.M = len(sequence)-1
    self.incr = incr    
    self.arithmetic = sequence[-1]
  
  def nextCashArmies(self):
    if self.cont <= self.M:
      armies = self.sequence[self.cont]
    else:
      self.arithmetic += self.incr
      armies = self.arithmetic
    self.cont += 1
    return armies


class Card(object):
  '''! Represents a territory card
  Has a code corresponding to the territory, and a kind (wildcard, soldier, horse, cannon in the original game)
  '''
  def __init__(self, code, kind):
    if kind == 0 and code >= 0:
      raise Exception(f"Wildcard (kind=0) can not have non negative code ({code})")
    self.code = code
    self.kind = kind
  
  def __eq__(self, o):
    '''! Wildcards are equal to any card, except other wildcards
    '''
    if self.kind==0 and o.kind==0:
      return False
    if self.kind==0 or o.kind==0:
      return True
    return self.kind == o.kind
    
  def __ne__(self, o):
    '''! Wildcards are different from any card, except other wildcards
    '''
    if self.kind==0 and o.kind==0:
      return False
    if self.kind==0 or o.kind==0:
      return True
    return self.kind != o.kind
    
  def __hash__(self):
    '''! The deck class that builds the deck of cards guarantees codes are unique. Wildcards are given negative numbers without repetitions
    '''
    return self.code
  
  def __repr__(self):
    return str(self.code)

class Deck(object):
  '''! Deck of territory cards with methods to find cashable sets
  '''
  def __init__(self):
    self.deck = []
  
  def shuffle(self):
    random.shuffle(self.deck)
  
  def create_deck(self, countries, num_wildcards = 2):
    '''! Create one card per country, and for the wildcards ensure their codes are negative and do not repeat
    '''
    for c in countries:
      self.deck.append(Card(c.code, c.card_kind))
    for i in range(num_wildcards):
      self.deck.append(Card(-i-1, 0))    
    self.orig_deck = [card for card in self.deck]
    self.shuffle()
    
  
  def draw(self):
    '''! Pop the first card of the deck.
    If deck is empty, we create a copy of the original one and reshuffle
    I will add the option to just stop giving cards once the deck is exhausted
    '''
    card = self.deck.pop(0)
    if len(self.deck)==0:
      self.deck = [card for card in self.orig_deck]
      self.shuffle()
    return card
  
  def __deepcopy__(self, memo):
    new_Deck = Deck()
    new_Deck.deck = copy.deepcopy(self.deck)
    new_Deck.orig_deck =  copy.deepcopy(self.orig_deck)
    return new_Deck
  
  @staticmethod
  def isSet(c1,c2,c3):
    '''! Tells if the three cards are all the same or all different. This includes the case where there is one wildcard
    '''
    if c1==c2 and c2==c3 and c1==c3: 
      return True
    if c1!=c2 and c2!=c3 and c1!=c3: 
      return True
    return False
  
  @staticmethod
  def containsSet(deck):
    '''! Tells if the given list of cards contains a set by iterating over all possible combinations of three cards
    '''
    if len(deck)<3: return False
    for c in itertools.combinations(deck, 3):
      if Deck.isSet(*c): return True
    return False
  
  @staticmethod
  def yieldCashableSet(deck):
    '''! Yield the first found cashable set. The order of visit is the one given by itertools.combinations
    '''
    if len(deck)<3: return None
    for c in itertools.combinations(deck, 3):
      if Deck.isSet(*c): return c
    return None
  
  @staticmethod
  def yieldBestCashableSet(deck, player_code, countries):
    '''! Yield the best found cashable set. Taking into account the 2 bonus armies received if the player owns a territory from a cashed card.
    '''
    best, best_s = None, -1
    if len(deck)<3: return None    
    for c in itertools.combinations(deck, 3):
      s = 0
      if Deck.isSet(*c):
        for card in c:
          if card.code >= 0 and countries[card.code].owner == player_code:            
            s += 1
        if best_s < s:
          best_s = s
          best = c        
    return best