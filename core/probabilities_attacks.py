# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:05:19 2021

@author: lucas
"""


import numpy as np
import pandas as pd

if False:
  nodes = 4
  R = 150
  theta_offset = np.pi
  H_offset, W_offset = 720/2,1280/2
  
  
  theta = 0
  theta_inc = np.pi * 2 / nodes
  for i in range(nodes):  
    x = round(np.cos(theta + theta_offset)*R + W_offset,0)
    y = round(np.sin(theta + theta_offset)*R + H_offset,0)
    print(f"({x: >5}, {y: >5})")
    theta += theta_inc
  
  
  
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_test.json"
  
from board import Board, World , RandomAgent   
import misc
    
# Create the net using the same parameters
inputs = misc.read_json(EI_inputs_path)
model_args =  misc.read_json(inputs["model_parameters"])
board_params = inputs["board_params"]
path_board = board_params["path_board"]
world = World(path_board)


# Set players
pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Blue'), RandomAgent('Green')
players = [pR1, pR2]
# Set board
prefs = board_params
        
board = Board(world, players)
board.setPreferences(prefs)



if False:
  num_tries = 1
  results = {(0,2):0, (1,1):0, (2,0):0}
  for _ in range(num_tries):
    aLoss, dLoss = board.roll(3, 2)
    results[(aLoss, dLoss)]+=1
  
  for k in results:
    results[k] /= num_tries
  
  print("Results of one attack")
  print(results)
  print("\n\n")

def attack_till_dead(A,D):
  while A>1 and D>0:
     aLoss, dLoss = board.roll(min(A-1,3), min(D,2))
     A -= aLoss
     D -= dLoss
  if A == 1:
    return 0
  elif D==0:
    return 1
  else:
    return None
if True:

  num_tries = 100000
  max_A, max_D = 20 , 20 
  res = [[0 for _ in range(3,max_D+1)] for _ in range(3,max_A+1)]
  for a in range(3,max_A+1):
    for d in range(3,max_D+1):
      aux = 0
      for _ in range(num_tries):
        aux += attack_till_dead(a, d)
        
      res[a-1][d-1] = aux/num_tries
      
      
  pd.DataFrame(res).to_csv("attack_probs.csv")
        
      