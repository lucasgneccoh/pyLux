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
from model import GCN_risk, load_dict, boardToData
import agent
import copy
import torch_geometric


# %% See policy for country pick over time
path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_04_09_test_map/models"
path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_02_09_test_map/models"
path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_07_09_classic/models"
path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_hex/models"

EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_test.json"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_test.json"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_classic.json"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_hex.json"


load_model = True
model_name = "model_6_0_initialFortify.tar"
model_name = "model_26_4_initialFortify.tar"
model_name = "model_8_2_initialFortify.tar"
model_name = "model_102_3_attack.tar" # Hex

# Create the net using the same parameters
inputs = misc.read_json(EI_inputs_path)
model_args = misc.read_json(inputs["model_parameters"])
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

# Load model
state_dict = load_dict(os.path.join(path_model, model_name),
                       device='cpu', encoding='latin1')
net.load_state_dict(state_dict['model'])
net.eval()
apprentice = agent.NetApprentice(net)


# Prepare board
board = copy.deepcopy(board_orig)

# board.playMove(agent.buildMove(board, ("pi", 1)))
# board.playMove(agent.buildMove(board, ("pi", 2)))
# board.playMove(agent.buildMove(board, ("pi", 3)))
# board.playMove(agent.buildMove(board, ("pi", 4)))
# board.playMove(agent.buildMove(board, ("pi", 5)))
# board.playMove(agent.buildMove(board, ("pi", 0)))




# board.playMove(agent.buildMove(board, ("pl", 1)))
# board.playMove(agent.buildMove(board, ("pl", 0)))

# board.playMove(agent.buildMove(board, ("pl", 1)))
# board.playMove(agent.buildMove(board, ("pl", 0)))

# board.playMove(agent.buildMove(board, ("pl", 1)))

# Play moves if needed
# board.playMove(agent.buildMove(board, ("pi", 0)))
# board.playMove(agent.buildMove(board, ("pi", 8)))
# board.playMove(agent.buildMove(board, ("pi", 1)))
# board.playMove(agent.buildMove(board, ("pi", 7)))
# board.playMove(agent.buildMove(board, ("pi", 2)))
# board.playMove(agent.buildMove(board, ("pi", 6)))
# board.playMove(agent.buildMove(board, ("pi", 3)))
# board.playMove(agent.buildMove(board, ("pi", 5)))
# board.playMove(agent.buildMove(board, ("pi", 4)))

# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))
# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))
# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))
# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))
# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))
# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))
# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))
# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("pl", 4)))


# Modifications for the attack example (Dont put them in SEU)
# board.world.countries[4].armies = 12
# board.world.countries[5].armies = 12


# # Modifications for the attack example (pinned in Peru)

# board.playMove(agent.buildMove(board, ("pl", 5)))
# board.playMove(agent.buildMove(board, ("a", -1,-1)))
# board.playMove(agent.buildMove(board, ("f", -1,-1)))

# board.world.countries[4].armies = 1
# board.world.countries[5].armies = 1


# board.world.countries[0].owner = 1
# board.world.countries[1].owner = 1
# board.world.countries[3].owner = 1
# board.world.countries[4].owner = 1

# board.world.countries[0].armies = 20 # Venezuela
# board.world.countries[1].armies = 1 # Brazil


# board.playMove(agent.buildMove(board, ("pl", 2)))

# board.world.countries[2].armies = 20



#%%%
board.play() 
while not board.gameOver and board.gamePhase != "attack":
  board.play()
  
  
board.report()
print(board.countriesPandas())
print("\n")

# Get policy for board
canon, _ = board.toCanonical(board.activePlayer.code)
batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
mask, moves = agent.maskAndMoves(canon, canon.gamePhase, batch.edge_index)
policy, value = apprentice.getPolicy(canon)
pop = policy.squeeze()

T = 1
exp = np.exp(np.log(np.maximum(pop, 0.000001))/T)
soft = exp/exp.sum()

co = board.countries()
for m, a, p, s in zip(mask.squeeze(), moves, pop, soft):
    if m.item():
        if len(a) > 2:
            print(
                f"{a[0]}: {co[a[1]]['id']} -> {co[a[2]]['id']} - {p:.3f} - {s:.3f}")
        else:
            print(f"{a[0]}: {co[a[1]]['id']} - {p:.3f}- {s:.3f}")






