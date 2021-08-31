# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 19:52:33 2021

@author: lucas
"""

#%%% Preliminaries

import agent

import numpy as np


import misc

import copy

import os

from board import Board, RandomAgent
from world import World

import torch
import torch_geometric
# Needs torch_geometric
from model import GCN_risk, load_dict, save_dict, boardToData



# Define inputs here
load_model = True
max_depth = 150
move_type = "startTurn"
path_data = "../data_diamond"
path_model = "../data_diamond/models"
model_name = "model_3_4_attack.tar"
num_sims = 600
saved_states_per_episode=1
verbose = 1
num_states = 4
batch_size = 4
temp = 1

EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_small.json"


# Create folders if they dont exist
move_types = ["initialPick", "initialFortify", "startTurn", "attack", "fortify"]
for folder in move_types:
    os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
os.makedirs(path_model, exist_ok = True)

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

##### Create the net 

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

# Load model

print(f"Chosen model is {model_name}")
state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')
# print(state_dict)
net.load_state_dict(state_dict['model'])
print("Model has been loaded\n\n")


##### Load the model to use as apprentice
net.eval()
apprentice = agent.NetApprentice(net)


# Start play
board = copy.deepcopy(board_orig)


#%%% Random play
#### Random play
for _ in range(0):
    board.play()

#%%% Board status
### See board status
board.report()
print(board.countriesPandas())



#%%% Human move

# See legal moves
print("\n*** Human info")
moves = board.legalMoves()
print(moves)


# Choose a move
m = moves[-1]
print("Human playing move: ", m)
board.playMove(m)


#%%% Net move
canon, map_to_orig = board.toCanonical(board.activePlayer.code)
batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
mask, moves = agent.maskAndMoves(canon, canon.gamePhase, batch.edge_index)

policy, value = apprentice.getPolicy(canon)

print("\n*** Net info")
print(mask)
print(moves)
print(policy)
print(value)
pol = policy.ravel() / policy.sum()
ind = np.random.choice(len(moves), p = pol)
m = agent.buildMove(board, moves[ind])
print("Net playing move: ", m)
board.playMove(m)


    
    
    
    
    
    
    
    
    

