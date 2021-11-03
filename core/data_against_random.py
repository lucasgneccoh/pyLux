# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 18:11:40 2021

@author: lucas
"""
import numpy as np
import pandas as pd
import misc
import os
import torch
from board import Board, World, RandomAgent
from model import GCN_risk, load_dict, boardToData
import agent
import copy
import torch_geometric
import re

def get_model_order(path):    
  
    # models = filter(lambda s: "initialPick" in s, os.listdir(path))
    models = filter(lambda s: True, os.listdir(path))
    
    def key_sort(x):
        s = x.split("_")
        r = None
        try:
            r = int(s[1])*100 + int(s[2])        
        except:
            r = -1
        return r
        
    return sorted(models, key=key_sort)





#%% LOAD MODEL

path_model = "../data_hex/models"
out_path = "../data_hex"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_hex.json"


# path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_01_09_test_map/models"
# # path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_test_git/models"
# EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_test_2.json"

# path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_07_09_classic/models"
# EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_classic.json"


load_model = True
model_name = "model_0_0_initialFortify.tar"

# Create the net using the same parameters
inputs = misc.read_json(EI_inputs_path)
model_args =  misc.read_json(inputs["model_parameters"])
board_params = inputs["board_params"]
path_board = board_params["path_board"]

# ---------------- Model -------------------------

print("Creating board")


world = World(path_board)


# Set players
pR1, pR2, pR3 = RandomAgent('Red'), RandomAgent('Blue'), RandomAgent('Green')
players = [pR1, pR2]
# Set board
prefs = board_params
        
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
# net.eval()


board = copy.deepcopy(board_orig)

#%%% GET DATA WIN RATIO

if True:

  max_turns = 150
  num_matchs = 100
  match_number = 4
  title_win_ratio = f"Hex map: win, loss and draws"
  
  labelsize = 17
  fontsize_ticks = 17
  fontsize_legend = 15
  fontsize_title = 20
  
  
  # Get data now
  models_sorted = get_model_order(path_model)
  wins = []
  netPlayer_countries = []
  netPlayer_armies = []
  op_countries = []
  op_armies = []
  model_cont = []
  for i, model_name in enumerate(models_sorted):
      a = re.search(f"[a-z]+_[0-9]+_{match_number}",model_name)
      a = 1
      if a is None: continue
      print(f"Chosen model is {model_name}")
      state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')    
      net.load_state_dict(state_dict['model'])
      net.eval()    
      for k in range(num_matchs):
        if (k+1)%10== 0: print(f'Match {k+1}')
        world = World(path_board)
        apprentice = agent.NetApprentice(net) 
        netPlayer = agent.NetPlayer(apprentice, move_selection = "random_proportional", temp = 0.5)
        # Play against random
        pRandom = RandomAgent('Random')
        battle_board = Board(world, [netPlayer, pRandom])
        battle_board.setPreferences(prefs)
        for j in range(max_turns):
          battle_board.play()
          if battle_board.gameOver: break
        
        w = 0
        if battle_board.players[netPlayer.code].is_alive:
          if not battle_board.players[pRandom.code].is_alive:
            w = 1
        else:
          w = -1
        wins.append(w)
        netPlayer_countries.append(battle_board.getPlayerCountries(netPlayer.code))
        netPlayer_armies.append(battle_board.getPlayerArmies(netPlayer.code))
        op_armies.append(battle_board.getPlayerArmies(pRandom.code))
        op_countries.append(battle_board.getPlayerCountries(pRandom.code))
        model_cont.append(i)
  
  
  
  data = pd.DataFrame(data = {
    "id":model_cont,
    "result":wins,
    "countries":netPlayer_countries,
    "armies":netPlayer_armies,
    "armies_op":op_armies,
    "countries_op":op_countries
    })
  
  data['win'] = data['result']==1
  data['loss'] = data['result']==-1
  data['draw'] = data['result']==0
  data['armies_win'] = data['armies'].where(data['win'], other = np.nan)
  data['armies_loss'] = data['armies'].where(data['loss'], other = np.nan)
  data['armies_draw'] = data['armies'].where(data['draw'], other = np.nan)
  
  data['armies_op_win'] = data['armies_op'].where(data['win'], other = np.nan)
  data['armies_op_loss'] = data['armies_op'].where(data['loss'], other = np.nan)
  data['armies_op_draw'] = data['armies_op'].where(data['draw'], other = np.nan)
  
  data.to_csv(os.path.join(out_path, "data_vs_random.csv"))
  


