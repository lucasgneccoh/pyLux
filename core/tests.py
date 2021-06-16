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
# from mcts import MCTS, MctsApprentice, NetApprentice
# import model

import json

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
  
  with open("../support/exp_iter_inputs/exp_iter_inputs.json",'r') as f:        
    inputs = json.load(f)
  
  print(inputs)
      
  
    

