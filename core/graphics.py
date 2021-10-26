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
path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_hex_final/models"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_hex.json"


path_model = "C:/Users/lucas/OneDrive/Documentos/stage_risk/data_02_09_test_map/models"
EI_inputs_path = "../support/exp_iter_inputs/exp_iter_inputs_test_2.json"


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
# net.eval()




#%%% MODIFICATIONS
# Prepare board
board = copy.deepcopy(board_orig)
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
# board.playMove(agent.buildMove(board, ("pl", 5)))


# Modifications for the attack example (pinned in Peru)

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


#%%%% GET DATA POLICIES
# Prepare table
countries_names = {x.code : x.id for x in board.countries()}

# Get data now
models_sorted = get_model_order(path_model)
policies = []
for i, model_name in enumerate(models_sorted):
    print(f"Chosen model is {model_name}")
    state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')    
    net.load_state_dict(state_dict['model'])
    net.eval()
    apprentice = agent.NetApprentice(net)    
    # Get policy for board
    canon, _ = board.toCanonical(board.activePlayer.code)
    batch = torch_geometric.data.Batch.from_data_list([boardToData(canon)])
    mask, moves = agent.maskAndMoves(canon, canon.gamePhase, batch.edge_index) 
    policy, value = apprentice.getPolicy(canon)
    policies.append(policy)


data = pd.DataFrame(np.array(policies).squeeze(1), index = models_sorted)

if len(data.columns) == len(countries_names):
  data.rename(columns = countries_names, inplace=True)
elif len(data.columns) == batch.edge_index.shape[1]+1:
  edge_names = {}
  for j in range(batch.edge_index.shape[1]):
    edge_names[data.columns[j]] = f"{countries_names[moves[j][1]]}_{countries_names[moves[j][2]]}"
  
  edge_names[data.columns[-1]] = "PASS"
  data.rename(columns = edge_names, inplace=True)
else:
  raise Exception("Algo raro")




#%%% Plot geenral parameters

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

roll = data.rolling(window=6).mean()



# ATTACK
# fig, ax = plt.subplots(1,1,figsize=(12,5))

# mask_s = mask.squeeze()
# for i, col in enumerate(roll):
#   if mask_s[i]:
#     if data[col].max()>0.25:
#       ax.plot(roll[col], label = col)

# ax.legend(loc='best', ncol=4, fancybox=True, shadow=True)
# ax.set_xlabel("Training step")
# ax.set_ylabel("Probability")
# ax.set_title(f"Eval attack")
# # bbox_to_anchor=(0.5, 1.05)



# PLACE

labelsize = 17
fontsize_ticks = 17
fontsize_legend = 15
fontsize_title = 20

plt.rc('xtick', labelsize=labelsize)
plt.rc('ytick', labelsize=labelsize)

fig, ax = plt.subplots(1,1,figsize=(12,5))

for col in roll:
  ax.plot(roll[col].to_numpy(), color = style[col], label = col)

ax.legend(loc='best', ncol=3, fancybox=True, shadow=True, fontsize=fontsize_legend)
ax.set_xlabel("Training step", fontsize=fontsize_ticks )
ax.set_ylabel("Probability", fontsize=fontsize_ticks )
ax.set_title(f"Hex map: Country draft in empty map", fontsize=fontsize_title)
# bbox_to_anchor=(0.5, 1.05)


plt.show()





#%%% GET DATA WIN RATIO

max_turns = 150
num_matchs = 50


# Get data now
models_sorted = get_model_order(path_model)
wins = []
netPlayer_countries = []
netPlayer_armies = []
model_cont = []
for i, model_name in enumerate(models_sorted):
    print(f"Chosen model is {model_name}")
    state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')    
    net.load_state_dict(state_dict['model'])
    net.eval()
    apprentice = agent.NetApprentice(net) 
    netPlayer = agent.NetPlayer(apprentice, move_selection = "random_proportional", temp = 1)
    # Play against random
    pRandom = RandomAgent('Random')
    for k in range(num_matchs):
      if (k+1)%%10== 0: print(f'Match {k+1}')
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
      model_cont.append(i)

data = pd.DataFrame(np.array(policies).squeeze(1), index = models_sorted)




