import itertools
from string import ascii_lowercase

from deck import Deck
from move import Move
import copy


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)
  
    
    
# agent.py
# I needed to put this here so that there were no cycles in the imports
# This way, agent.py can use it and also board.py without having problems

