import agent
from board import Board
from world import World, Country, Continent
from move import Move
from mcts import MCTS, MctsApprentice, NetApprentice, maskAndMoves, buildMove
from model import boardToData, GCN_risk, RiskDataset, saveBoardObs

import os
import itertools
import numpy as np
import copy
import sys
import json

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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--inputs", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/exp_iter_inputs.json")
  args = parser.parse_args()
  return args
  
def read_json(path):
  with open(path, 'r') as f:
    data = json.load(f)
  return data
    

def print_message_over(message):
    sys.stdout.write('\r{0}'.format(message))
    sys.stdout.flush()

def build_expert_mcts(apprentice):
    return MCTS(root=None, apprentice=apprentice, max_depth = 50, 
                sims_per_eval = 5, num_MCTS_sims = 10000,
                wa = 10, wb = 10, cb = np.sqrt(2))

def isint(n):
    try:
        x = int(n)
        return True
    except:
        return False
        
def create_self_play_data(path, root, num_samples, start_sample, apprentice, expert, max_depth = 100, saved_states_per_episode=1, verbose = False):
    """ Function to create episodes from self play.
        Visited states are saved and then re visited with the expert to label the data

    """
    samples = 0

    samples_type = {'initialPick':0, 'initialFortify':0, 'startTurn':0, 'attack':0, 'fortify':0}
    for k, v in samples_type.items():
        path_aux = os.path.join(path, k, 'raw')
            
        val = max(list(map(int,
                        filter(isint,
                                    [n[(n.find("_")+1):n.find(".")] 
                                        for n in  os.listdir(path_aux) if 'board' in n]
                                )
                        )
                        ) + [0])
        samples_type[k] = val

    move_to_save = itertools.cycle(list(samples_type.keys()))
    edge_index = boardToData(root).edge_index
    while samples<num_samples:     

        # ******************* PLAY EPISODE ***************************
        episode = []
        state = copy.deepcopy(root)
        for i in range(max_depth):  
            print_message_over(f"Playing episode: {i}/{max_depth}")

            # Check if episode is over            
            if state.gameOver: break

            # Check is current player is alive or not
            if not state.activePlayer.is_alive: 
                # print("\npassing, dead player")
                state.endTurn()
                continue

            # Get possible moves, and apprentice policy
            mask, actions = maskAndMoves(state, state.gamePhase, edge_index)
            try:
                policy, value = apprentice.play(state)
            except Exception as e:
                state.report()
                print(state.activePlayer.is_alive)
                print(state.activePlayer.num_countries)
                raise e
            policy = policy * mask
            probs = policy.squeeze().detach().numpy()
            probs =  probs / probs.sum()

            ind = np.random.choice(range(len(actions)), p = probs)
            move = buildMove(state, actions[ind])
            
            episode.append(copy.deepcopy(state))

            # Play the move to continue
            state.playMove(move)
           
            
        # ******************* SAVE STATES ***************************
        # Take some states from episode
        # Choose which kind of move we are going to save
        
        to_save = next(move_to_save)
        
        try:
            # Define here how many states to select, and how
            options = [s for s in episode if s.gamePhase == to_save]
            init_to_save = to_save
            while not options:
                to_save = next(move_to_save)
                if to_save == init_to_save:
                    raise Exception("Episode is empty? No dataset could be created for any game phase")
                options = [s for s in episode if s.gamePhase == to_save]
            states_to_save = np.random.choice(options, min(saved_states_per_episode, len(options)))
        except Exception as e:
            raise e

        # Get expert move for the chosen states
        for i, state in enumerate(states_to_save):
            print_message_over(f"Saving states: Saved {i}/{len(states_to_save)}... Total: {samples}/{num_samples}")
            policy_exp, value_exp, _ = expert.getActionProb(state, temp=1, num_sims = None, use_val = False)
            # Save the board, value and target
            board, _ = state.toCanonical(state.activePlayer.code)
            phase = board.gamePhase
            if isinstance(policy_exp, torch.Tensor):
                policy_exp = policy_exp.detach().numpy()
            if isinstance(value_exp, torch.Tensor):
                value_exp = value_exp.detach().numpy()
            
            saveBoardObs(path + '/' + phase + '/raw', 'board_{}.json'.format(samples_type[phase]),
                            board, board.gamePhase, policy_exp.tolist(), value_exp.tolist())
            samples += 1
            samples_type[phase] += 1
            print_message_over(f"Saving states: Saved {i+1}/{len(states_to_save)}... Total: {samples}/{num_samples}")
            
    print_message_over("Done!")
    print()


def TPT_Loss(output, target):
    return -(target*torch.log(output)).sum()


# ---------------- Start -------------------------
args = parseInputs()
inputs = read_json(args.inputs)


iterations = inputs["iterations"]
num_samples = inputs["num_samples"]
saved_states_per_episode = inputs["saved_states_per_episode"]
max_depth = inputs["max_depth"]
initial_apprentice_mcts_sims = inputs["initial_apprentice_mcts_sims"]
expert_mcts_sims = inputs["expert_mcts_sims"]

path_data = inputs["path_data"]
path_model = inputs["path_model"]
batch_size = inputs["batch_size"]
model_args =  read_json(inputs["model_parameters"])

path_board = inputs["path_board"]

# ---------------- Model -------------------------




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

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = TPT_Loss



move_types = ['initialPick', 'initialFortify', 'startTurn',
                                'attack', 'fortify']
                                
                                
                                
# Define initial apprentice
apprentice = MctsApprentice(num_MCTS_sims = initial_apprentice_mcts_sims, temp=1, max_depth=max_depth)
# apprentice = NetApprentice(net)

# build expert
expert = build_expert_mcts(None) # Start with only MCTS with no inner apprentice
expert.num_MCTS_sims = expert_mcts_sims
                     
# Create folders to store data
for folder in move_types:
    os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
os.makedirs(path_model, exist_ok = True)
                                
from random import shuffle

state = copy.deepcopy(board_orig)
state.initialPhase = True
state.pickInitialCountries = True
# state.play() # random init
for i in range(iterations):
    # Sample self play
    # Use expert to calculate targets
    create_self_play_data(path_data, state, num_samples, num_samples*i,
                          apprentice, expert, max_depth = max_depth,
                          saved_states_per_episode = saved_states_per_episode, verbose=True)

    # Train network on dataset
    shuffle(move_types)
    for j, move_type in enumerate(move_types):
        save_path = f"{path_model}/model_{i}_{j}_{move_type}.tar"
        root_path = f'{path_data}/{move_type}'
        
        if len(os.listdir(os.path.join(root_path, 'raw')))<batch_size: continue
        
        risk_dataset = RiskDataset(root = root_path)
        # TODO: add validation data
        loader = G_DataLoader(risk_dataset, batch_size=batch_size, shuffle = True)
        train_model(net, optimizer, scheduler, criterion, device,
                    5, loader, val_loader = loader, eval_every = 1,
                    load_path = None, save_path = save_path)

    # build expert with trained net
    expert = build_expert_mcts(NetApprentice(net))
