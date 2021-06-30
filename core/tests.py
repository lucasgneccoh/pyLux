# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:52:33 2021

@author: lucas
"""


from world import World, Country, Continent
from board import Board
import agent
import time
import numpy as np
import copy
import model

# from mcts import MCTS, MctsApprentice, NetApprentice
# import model

import json
import os



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch_geometric 
from torch_geometric.data import DataLoader as G_DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.data import Dataset as G_Dataset
from torch_geometric.data import download_url
import torch_geometric.transforms as T
from torch_geometric import utils



def func_to_par(t):
    time.sleep(t)
    #return t*t
    os.makedirs('../data/test_par/dir_{}'.format(t), exist_ok = True)
    return True


#%% TESTING
if __name__ == '__main__':

  console_debug = True
  
  # Load map
  path = '../support/maps/classic_world_map.json'
  path = '../support/maps/test_map.json'
    
  world = World(path)
  

  # Set players
  aggressiveness = 0.7
  pH, pR1, pR2 = agent.Human('PocketNavy'), agent.RandomAgent('Red',aggressiveness), agent.RandomAgent('Green',aggressiveness)
  pMC = agent.FlatMC('MC', agent.RandomAgent, budget=300)
  pMC2 = agent.FlatMC('MC', agent.RandomAgent, budget=300)
  pP = agent.PeacefulAgent()
  pA = agent.RandomAggressiveAgent('RandomAgressive')
 
  players = [pR1, pA]
  # Set board
  prefs = {'initialPhase': False, 'useCards':True,
           'transferCards':True, 'immediateCash': True,
           'continentIncrease': 0.05, 'pickInitialCountries':True,
           'armiesPerTurnInitial':4,'console_debug':False}  
  board_orig = Board(world, players)
  board_orig.setPreferences(prefs)
  board_orig.showPlayers()
  
  #%% Test play
  if False:
    print("\nTest play\n")
    import tqdm
    board = copy.deepcopy(board_orig)
    N = 20
    start = time.process_time()
    iter_times = []
    armies = []
    countries = []
    board.console_debug = True
    
    pMC.budget = 500
    pMC.inner_placeArmies_budget = 100
    pMC.console_debug = True
    
    for i in range(N):
      in_start = time.process_time()
      board.play()
      board.report()
      print('-'*30)
      
      user_msg = input("Press q to quit, c to continue: ")
      while not user_msg in ['q', 'c']:
        user_msg = input("Press q to quit, c to continue: ")
      if user_msg == 'q': break
      print('-'*30)
      iter_times.append(time.process_time()-in_start)
      armies.append(board.getAllPlayersArmies())
      countries.append(board.getAllPlayersNumCountries())
      if board.gameOver: break
    total = time.process_time()-start
    
    armies = np.array(armies)
    countries = np.array(countries)
    
    print(f"Total time: {total} sec \t Avg time per play: {total/N}")
    board.report()
    
    import matplotlib.pyplot as plt
    for i, p in enumerate(players):
      plt.plot(armies[:,i], label=p.name())
    plt.legend()
    plt.title("Armies")
    plt.show()
    
    for i, p in enumerate(players):
      plt.plot(countries[:,i], label=p.name())
    plt.legend()
    plt.title("Countries")
    plt.show()
  
  #%% Test deepcopy 
  if False:
    print("\nTest deepcopy\n")
    board = copy.deepcopy(board_orig)
    board_copy = copy.deepcopy(board)
    for i in range(10):
      board.play()
      board_copy.play()
    board.report()
    board_copy.report()
    for i in range(5):
      board_copy.play()
    print("\n\nSee one country")
    print(board.getCountryById('BRA'))
    print(board_copy.getCountryById('BRA'))
    
    print("\n\nWorld reference")
    print(board.world)
    print(board_copy.world)
    
    print("\n\nPlayers")
    for i, p in board.players.items():
      print(i, p)
      print('Cards:', p.cards)
    for i, p in board_copy.players.items():
      print(i, p)
      print('Cards:', p.cards)
      
      
    print("\n\nDecks")
    print("\nboard")
    for c in board.deck.deck:
      print(c)
    print("\nboard_copy")
    for c in board_copy.deck.deck:
      print(c)
      
    print("Draw one card", board.deck.draw())
    print("\n\nDecks")
    print("\nboard")
    for c in board.deck.deck:
      print(c)
    print("\nboard_copy")
    for c in board_copy.deck.deck:
      print(c)
  
  #%% Test simulate
  if False:
    print("\nTest simulate\n")
    pMC.console_debug = False
    players = [pMC, copy.deepcopy(pR2)]
    board = Board(copy.deepcopy(world), players)
    board.showPlayers()
    board.setPreferences(prefs)
    for i in range(2):
      board.play()
    board.report()

    for j in range(2):
      sim_board = copy.deepcopy(board)

      # newAgent (instance, not class), playerCode
      
      sim_board.simulate(agent.RandomAgent(),pMC.code, changeAllAgents=False,
                         maxRounds=20, safety = 10e5, sim_console_debug=False)
      print(f"\nBoard after simulation {j}\n")
      sim_board.report()
      print("\nScore for 0")
      pMC2.setPrefs(0)
      print(pMC2.score(sim_board))
      print("\nScore for 1")
      pMC2.setPrefs(1)
      print(pMC2.score(sim_board))
      
    
  
    
  #%% Test fromDicts, toDicts
  if False:
    print("\nTest fromDicts, toDicts\n")
    print('board_orig')
    board_orig.showPlayers()
    board = copy.deepcopy(board_orig)
    board.showPlayers()
    board.report()
    print('board')
    for i, p in board.players.items():
      print(i, p.code, p)
    for _ in range(3):
      board.play()
      
    print("\n\n Turning board to Dicts")
    continents, countries, inLinks, players, misc = board.toDicts()
    print(players)
    board_copy_1 = copy.deepcopy(board)
    board_copy_2 = Board.fromDicts(continents, countries, inLinks,\
                                 players, misc)
      
    print("Orig board:")
    print(board.countries())
    print("\nCopy 1")
    print(board_copy_1.countries())
    print("\nCopy 2")
    print(board_copy_2.countries())
    print()
    print(board.activePlayer.code)
    print(board_copy_2.activePlayer.code)

  #%% Dicts and hashes
  if False:
    print('\nTesting dicts and hashs\n')
    test_dict = {}
    test_dict[board] = {}
    c = board.countries()[0]
    move = agent.Move(c,c,1,'startTurn')
    test_dict[board][move] = 'testing'
    move = agent.Move(c,c,1,'attack')
    test_dict[board][move] = 'testing 2'
    
    test_dict[board] = 5
    board.gamePhase = 'other'
    test_dict[board] += 5
    print(test_dict)
    


  #%% Test model and self play
  if False:
    with open("../support/exp_iter_inputs/exp_iter_inputs.json",'r') as f:        
      inputs = json.load(f)
    
    print(inputs)


  # Multprocessing
  if False:
    from multiprocessing import Pool, cpu_count
    cpus = cpu_count()
    print("CPUs: ", cpus)
    
    inputs = [2,3,4,5]
    
    start = time.perf_counter()
    for t in inputs:
        print(func_to_par(t))
    end = time.perf_counter()
    print("SEQ: ", end - start)
    
    
    start = time.perf_counter()
    with Pool(4) as pool:
        print(pool.map(func_to_par, inputs))
    
    end = time.perf_counter()
    print("PAR: ", end - start)    
  

  # neuralMCTS test. It is failing to tag the moves in the expert iteration process
  if True:
    path_model = "../data/models"
    EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs.json"
    
    # Create the net using the same parameters
    inputs = read_json(EI_inputs_path)
    model_args =  read_json(inputs["model_parameters"])
    path_board = inputs["path_board"]

    # ---------------- Model -------------------------

    print("Creating board")

    #%%% Create Board
    world = World(path_board)


    # Set players
    pR1, pR2, pR3 = agent.RandomAgent('Red'), agent.RandomAgent('Blue'), agent.RandomAgent('Green')
    players = [pR1, pR2, pR3]
    # Set board
    # TODO: Send to inputs
    prefs = {'initialPhase': True, 'useCards':True,
            'transferCards':True, 'immediateCash': True,
            'continentIncrease': 0.05, 'pickInitialCountries':True,
            'armiesPerTurnInitial':4,'console_debug':False}
            
    board_orig = Board(world, players)
    board_orig.setPreferences(prefs)

    num_nodes = board_orig.world.map_graph.number_of_nodes()
    num_edges = board_orig.world.map_graph.number_of_edges()

    print("Creating model")
    net = GCN_risk(num_nodes, num_edges, 
                     model_args['board_input_dim'], model_args['global_input_dim'],
                     model_args['hidden_global_dim'], model_args['num_global_layers'],
                     model_args['hidden_conv_dim'], model_args['num_conv_layers'],
                     model_args['hidden_pick_dim'], model_args['num_pick_layers'], model_args['out_pick_dim'],
                     model_args['hidden_place_dim'], model_args['num_place_layers'], model_args['out_place_dim'],
                     model_args['hidden_attack_dim'], model_args['num_attack_layers'], model_args['out_attack_dim'],
                     model_args['hidden_fortify_dim'], model_args['num_fortify_layers'], model_args['out_fortify_dim'],
                     model_args['hidden_value_dim'], model_args['num_value_layers'],
                     model_args['dropout'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    # Choose a model at random
    model_name = np.random.choice(os.listdir(path_model))    
    state_dict = load_dict(os.path.join(path_model, model_name), device)
    net.load_state_dict(state_dict['model'])
    
    print("Model has been loaded")
    print(net)
    
    
    
    
    
    
    
    
    
    

