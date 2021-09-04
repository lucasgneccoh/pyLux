# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 18:11:40 2021

@author: lucas
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import misc
import os
import torch
from board import Board, World, RandomAgent
from model import GCN_risk, load_dict
import agent
import copy

def get_model_order(path):    
    models = os.listdir(path)
    
    def key_sort(x):
        s = x.split("_")
        r = None
        try:
            r = int(s[1])*100 + int(s[2])        
        except:
            r = -1
        return r
        
    return sorted(models, key=key_sort)





#%% See policy for country pick over time
path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_02_09_test_map/models"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_test.json"

load_model = True
model_name = "model_27_0_initialPick.tar"

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



# Prepare board
board = copy.deepcopy(board_orig)
# Play moves if needed


# Prepare table
countries_names = {x.code : x.id for x in board.countries()}

# Get data now
models_sorted = get_model_order(path_model)
policies = []
for i, model_name in enumerate(models_sorted):
    print(f"Chosen model is {model_name}")
    state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')    
    net.load_state_dict(state_dict['model'])
    apprentice = agent.NetApprentice(net)    
    # Get policy for board
    canon, _ = board.toCanonical(board.activePlayer.code)
    policy, value = apprentice.getPolicy(canon)
    policies.append(policy)
    
data = pd.DataFrame(np.array(policies).squeeze(1))

data.rename(columns = countries_names, inplace=True)
ax = data.rolling(window=5).mean().plot.line()
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
plt.show()