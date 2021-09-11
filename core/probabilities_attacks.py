# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:05:19 2021

@author: lucas
"""


import numpy as np
import pandas as pd

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
  
def attack_till_dead_a_left(A,D):
  while A>1 and D>0:
     aLoss, dLoss = board.roll(min(A-1,3), min(D,2))
     A -= aLoss
     D -= dLoss
  return A
  
import time
def matrix(a, d, num_tries):
  r = np.array([[0,0],[0,0]])
  for _ in range(num_tries):
    r[0,0] += attack_till_dead(a, a)
    r[0,1] += attack_till_dead(a, d)
    r[1,0] += attack_till_dead(d, a)
    r[1,1] += attack_till_dead(d, d)
    
  r = r / num_tries
  return r.round(3)

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




if False:

  num_tries = 100000
  max_A, max_D = 20 , 20 
  res = [[0 for _ in range(3,max_D+1)] for _ in range(3,max_A+1)]
  for i, a in enumerate(range(3,max_A+1)):
    for j, d in enumerate(range(3,max_D+1)):
      aux = 0
      for _ in range(num_tries):
        aux += attack_till_dead(a, d)
      res[i][j] = aux/num_tries
      
      
  pd.DataFrame(res).to_csv("attack_probs.csv")

def expected_armies_remain(A,D, num_tries = 2000):
  res = attack_till_dead_a_left(A,D)
  for i in range(num_tries-1):
    res = (res*(i+1) + attack_till_dead_a_left(A, D))/(i+2)
  print(f"Expected number of armies left for A after battle (A = {A}, D = {D}): {res} ")

  

      
if False:
  A = 12
  D = 1
  p_a = 855/1296

  for a in range(A, -1, -1):
    print(a)
  
  print(f"Expected number of armies left for A after battle (A = {A}, D = {D}): {res} ")
  
if False:

  
  start = time.process_time()  
  num_tries = 20000
  
  a, d = 38,28
  start = time.process_time()
  r = matrix(a,d, num_tries)
  end = time.process_time()
  print(f"A\D: A = {a}, D = {d}")
  print(r)
  print(f"Time taken: {end - start}\n")
  
  
  
  a, d = 28,38
  start = time.process_time()
  r = matrix(a,d, num_tries)
  end = time.process_time()
  print(f"A\D: A = {a}, D = {d}")
  print(r)
  print(f"Time taken: {end - start}")
  

class Node(object):
  
  def is_valid(node):
    return node.a >= 1 and node.d >= 0
  
  
  def __init__(self, a,d):
    self.a = a
    self.d = d
  
  
  def get_children(self):
    if self.a==1 or self.d==0: return []
    
    a, d = self.a, self.d
    
    if min(self.a-1, self.d) == 1:
      res = [Node(a-1,d), Node(a, d-1)]
    else: 
      res = [Node(a-2,d), Node(a-1, d-1), Node(a, d-2)]
 
    return filter(Node.is_valid, res)
  
  def get_transition(self, other):
    dice_a, dice_d = min(self.a-1,3), min(self.d, 2)
    outcome = None
    if self.a == other.a:
      outcome = 1
    elif self.d == other.d:
      outcome = -1
    elif self.a -1 == other.a and self.d-1 == other.d:
      outcome = 0
    else:
      return None

    return (dice_a, dice_d, outcome)
  
  def __hash__(self):
    return hash((self.a, self.d))
  
  def __repr__(self):
    return f"({self.a}, {self.d})"
  
  def __eq__(self, other):
    return self.a==other.a and self.d == other.d
  
  
  
  
class Tree(object):

  def is_leaf(node):
    return node.a == 1 or node.d == 0
  
  def eval_leaf(node):
    return 1 if node.a>1 else 0
  
  def __init__(self, T, probs={}):
    self.probs = {k: v for k, v in probs.items()}
    self.T = T
  
  def compute(self, node):
    # print("Computing for Node ", node)
    
    if Tree.is_leaf(node): 
      # print("Is leaf")
      return Tree.eval_leaf(node)
    
    r = self.probs.get(node)
    if not r is None:  return r
    
    child = node.get_children()
    
    # Child has the children nodes, compute for current
    # Use transitions
    s = 0
    for c in child:
      p = self.compute(c)
      t = node.get_transition(c)
      # print("Child: ", c, "\tTransition: ", t, "\tProb: ", p)
      s += self.T[t]*p
    
    self.probs[node] = s
    return s
      
    
  def build_tree(self, a, d):
    node = Node(a,d)
    r = self.compute(node)
    return r
    
  
if False:
  T = {(1,1,1): 540/1296,
       (1,1,-1): 756/1296,
       (2,1,1): 750/1296,
       (2,1,-1): 546/1296,
       (3,1,1): 855/1296,
       (3,1,-1): 441/1296,
       (1,2,1): 330/1296,
       (1,2,-1): 996/1296,
       (2,2,1): 295/1296,
       (2,2,0): 420/1296,
       (2,2,-1): 581/1296,
       (3,2,1): 2890/7776,
       (3,2,0): 2611/7776,
       (3,2,-1): 2275/7776}
  
  # n = Node(2,1)
  # child = n.get_children()
  # print(child)
  # print(Node.is_valid(child[0]))
  # print(Node.is_valid(child[1]))
  
  tree = Tree(T, {})
  max_armies= 40
  p = tree.build_tree(max_armies, max_armies)
  p = tree.build_tree(max_armies-1, max_armies)
  p = tree.build_tree(max_armies, max_armies-1)

  res = [[0 for _ in range(max_armies+1)] for _ in range(max_armies+1)]
  for i, a in enumerate(range(max_armies+1)):
    for j, d in enumerate(range(max_armies+1)):
      r = tree.probs.get(Node(a,d))
      if r is None:
        res[a][d] = -1
      else:
        res[a][d] = r
        
  pd.DataFrame(res).to_csv("attack_probs_tree.csv")
  

  
if False:
  #%% Graphics
  
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.size': 20})
  # from scipy.interpolate import make_interp_spline, BSpline
  
  fig, ax = plt.subplots(1,1, figsize=(12,7))
  
  diag = np.array(list(filter(lambda x: x>=0, [res[i][i] for i in range(len(res))])))
  x_axis = np.arange(diag.shape[0])+2
  
  # xnew = np.linspace(x_axis.min(), x_axis.max(), 300) 
  # spl = make_interp_spline(x_axis, diag, k=3)  # type: BSpline
  # diag_smooth = spl(xnew)
  
  line_50 = [0.5 for _ in diag]
  
  
  
  ax.plot(x_axis, line_50, color="red", linestyle='-')
  ax.plot(x_axis, diag, color="blue")
  ax.scatter(x_axis, diag, color="blue")
  ax.set_xlabel("Number of armies for both A and D")
  ax.set_xticks(x_axis)
  ax.set_ylabel("Probability")
  ax.set_title("Probability of winning a battle as the attacker with same number of troops\n")
  plt.show()
  
  
  # Valor de las armadas
  diff = -10
  prob_att = []
  prob_def = []
  
  lim_inf = 2 if diff>0 else 2-diff
  lim_sup = max_armies-diff if diff>0 else max_armies
  x_axis = range(lim_inf, lim_sup)
  for a in x_axis:
    d = a + diff
    prob_att.append(tree.probs[Node(a,d)])
    prob_def.append(tree.probs[Node(d,a)])
  
  att = np.array(prob_att)
  defe = 1 - np.array(prob_def)
  
  
  fig, ax = plt.subplots(1,1, figsize=(20,9))
  
  ax.plot(x_axis, att, color="red", linestyle='-', label="Attacking")
  ax.plot(x_axis, defe, color="blue", label="Defending")
  ax.set_xlabel("Armies for player")
  ax.set_xticks(list(filter(lambda x: x%2==0, x_axis)))
  ax.set_ylabel("Probability")
  sign = "+" if diff>0 else "-"
  ax.set_title(f"Probability of winning a battle having x armies against {sign}{abs(diff)}")
  ax.legend(loc="best")
  plt.show()
    
    
    
    
    
#%% Brute force
import time
start = time.process_time()
res = {(1,1):0, (0,2):0, (2,0):0}
for a1 in range(6):
  for a2 in range(6):
    for a3 in range(6):
      for d1 in range(6):
        for d2 in range(6):
          A = sorted([a1,a2,a3], reverse=True)
          D = sorted([d1,d2], reverse=True)
          aLoss, dLoss = 0,0
          for x,y in zip(A,D):
            if x<=y:
              aLoss += 1
            else:
              dLoss += 1
          res[(aLoss, dLoss)] += 1
end= time.process_time()
print(f"Time taken:: {end-start}")
print(res)
          
          
     
    
    
    