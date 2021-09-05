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





#%% See policy for country pick over time
path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_hex/models"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_hex.json"

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
board.playMove(agent.buildMove(board, ("pi", 5)))
board.playMove(agent.buildMove(board, ("pi", 4)))
board.playMove(agent.buildMove(board, ("pi", 3)))
board.playMove(agent.buildMove(board, ("pi", 2)))
board.playMove(agent.buildMove(board, ("pi", 1)))
board.playMove(agent.buildMove(board, ("pi", 0)))

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


style = {"C0":(252/255, 59/255, 45/255), 
         "C1":(140/255, 12/255, 3/255),
         "C2":(255/255, 176/255, 31/255),
         "C3":(240/255, 128/255, 24/255),
         "C4":(23/255, 236/255, 255/255),
         "C5":(37/255, 84/255, 250/255),
         "VEN":(255/255, 0/255, 0/255),
         "BRA":(255/255, 80/255, 80/255),
         "PER":(255/255, 160/255, 160/255),
         "NAF":"black",
         "EGY":(160/255, 160/255, 160/255),
         "MEX":(255/255, 145/255, 0/255),
         "GRE":(255/255, 191/255, 107/255),
         "ICE":(0/255, 251/255, 255/255),
         "SEU":(0/255, 42/255, 255/255),}

roll = data.rolling(window=8).mean()

fig, ax = plt.subplots(1,1,figsize=(12,5))

for col in roll:
  ax.plot(roll[col], color = style[col], label = col)

ax.legend(loc='best', ncol=3, fancybox=True, shadow=True)
ax.set_xlabel("Training step")
ax.set_ylabel("Probability")
ax.set_title("Test map: Pick country after ICE is not available")
# bbox_to_anchor=(0.5, 1.05)

plt.show()

