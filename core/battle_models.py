from board import Board
from world import World, Country, Continent
from move import Move
import agent
from model import load_checkpoint, GCN_risk
import misc

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

import pandas as pd

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def append_each_field(master, new):
    for k, v in new.items():
        if k in master:
            master[k].append(v)
        else:
            master[k] = [v]
    return master
    
def parseInputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputs", help="Json file containing the inputs for the battles to run", default = "../support/battles/test_battle.json")    
    args = parser.parse_args()
    return args 
    

def load_puct(board, args):
    num_nodes = board.world.map_graph.number_of_nodes()
    num_edges = board.world.map_graph.number_of_edges()
    model_args =  read_json(args["model_parameters_json"])
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
    
    state_dict = load_dict(args["model_path"], device = 'cpu', encoding = 'latin1')

    net.load_state_dict(state_dict['model'])   

    apprentice = NetApprentice(net)
             
    kwargs = {}
    for a in ["sims_per_eval", "num_MCTS_sims", "wa", "wb", "cb", "temp", "use_val"]:
        if a in args: kwargs[a] = args[a]
    pPUCT = PUCTPlayer(apprentice = apprentice, **kwargs)

    
    return pPUCT
    
def battle(args):
    results = {}
    # Create players and board
    board_params = args["board_params"]
    world = World(board_params["path_board"])

    list_players = []
    for i, player_args in enumerate(args["players"]):
        kwargs = removekey(player_args, "agent")
        if player["agent"] == "RandomAgent":
            list_players.append(agent.RandomAgent(f"Random_{i}"))
        elif player["agent"] == "PeacefulAgent":
            list_players.append(agent.PeacefulAgent(f"Peaceful_{i}"))
        elif player["agent"] == "FlatMCPlayer":
            list_players.append(agent.FlatMCPlayer(name=f'flatMC_{i}', **kwargs))
        elif player["agent"] == "UCTPlayer":
            list_players.append(agent.UCTPlayer(name=f'UCT_{i}', **kwargs))
        elif player["agent"] == "PUCTPlayer":            
            board = Board(world, [agent.RandomAgent('Random1'), agent.RandomAgent('Random2')])
            board.setPreferences(board_params)
            puct = load_puct(board, player_args)
            list_players.append(puct)

    # Baselines            
    board = Board(world, list_players)
    board.setPreferences(board_params)
    
    board.report()
    # TODO: FINISH HERE, play the game, get the results
    return {"num_players": len(board.players), "board_id": board.board_id}
  
if __name__ == '__main__':
    
    args = parseInputs()
    inputs = misc.read_json(args.inputs)
    
    print("Inputs")
    print(inputs)
    
        
    board_params = inputs["board_params"]
    battles = inputs["battles"]
          
    
    # Battle here. Create agent first, then set number of matches and play the games    
    results = {}
    for b_name, b_args in battles.items():
        
        res = battle(b_args)
        res['b_name'] = b_name
        results = append_each_field(results, res)
        
    # Write csv with the results
    csv = pd.DataFrame(data = results)
    csv.to_csv(f"../support/battles/{b_name}.csv")
    print("Battle done")
    
    

