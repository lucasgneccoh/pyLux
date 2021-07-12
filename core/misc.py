import itertools
from string import ascii_lowercase

from deck import Deck
from move import Move
import copy
import sys
import os
import json


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)
  
def print_message_over(message):
    sys.stdout.write('\r{0}'.format(message))
    sys.stdout.flush()
    
def print_and_flush(message, sep="\n"):
    sys.stdout.write(message + sep)
    sys.stdout.flush()
    
def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
  
def write_json(data, path):
    try:
        with open(path,'w') as f:
            json.dump(data, f)
        return True
    except Exception as e:
        raise e
        
        
def remove_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
        else:
            return False
    except Exception as e:
        raise e
     

    
 
