# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:05:19 2021

@author: lucas
"""


import numpy as np

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
  
  
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_small.json"

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

num_tries = 10000
def attack_till_dead(A,D):
  while A>0 and D>0:
     aLoss, dLoss = board.roll(min(A,3), min(D,2))
     A -= aLoss
     D -= dLoss
  if A == 0:
    return 0
  elif D==0:
    return 1
  else:
    return None

a_armies, d_armies = 20 , 8 

res = 0
for _ in range(num_tries):
  res += attack_till_dead(a_armies, d_armies)
  
print(f"Results of attack till dead A = {a_armies}, D = {d_armies}")
print(f"Attacker wins {res/num_tries} of the times")
print(f"Defender wins {1 - res/num_tries} of the times")
print()

aux = a_armies
a_armies = d_armies
d_armies = aux
res = 0
for _ in range(num_tries):
  res += attack_till_dead(a_armies, d_armies)
  
print(f"Results of attack till dead A = {a_armies}, D = {d_armies}")
print(f"Attacker wins {res/num_tries} of the times")
print(f"Defender wins {1 - res/num_tries} of the times")
print()



  
  