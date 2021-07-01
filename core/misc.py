import itertools
from string import ascii_lowercase

from deck import Deck
from move import Move
import copy
import sys


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)
  
  
def print_message_over(message):
    sys.stdout.write('\r{0}'.format(message))
    sys.stdout.flush()
    
 
