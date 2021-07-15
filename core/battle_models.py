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
    
def load_flat(args):
    return agent.FlatMCPlayer(name='flatMC', max_depth = int(args["max_depth"]),
            sims_per_eval = int(args["sims_per_eval"]),
            num_MCTS_sims = int(args["num_MCTS_sims"]), cb = 0)
    
def load_uct(args):
    return agent.UCTPlayer(name='UCT', max_depth = int(args["max_depth"]),
            sims_per_eval = int(args["sims_per_eval"]),
            num_MCTS_sims = int(args["num_MCTS_sims"]), cb = float(args["cb"]))
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
             
    pPUCT = PUCTPlayer(apprentice = apprentice, max_depth = int(args["max_depth"]),
            sims_per_eval = int(args["sims_per_eval"]),
            num_MCTS_sims = int(args["num_MCTS_sims"]),
            wa = float(args["10"]), wb = float(args["10"]), cb = float(args["cb"]),
            temp = float(args["1"]),
            use_val = float(args["use_val"])
            )

    
    return pPUCT
    
def battle(args):
    results = {}
    # TODO: Finish this function, then go to the main

    return results
  
if __name__ == '__main__':
    
    args = parseInputs()
    inputs = misc.read_json(args.inputs)
        
    board_params = inputs["board_params"]
    battles = inputs["battles"]
    
    world = World(board_params["path_board"])
    
    

    ###### Set players  
    # Baselines
    pRandom = agent.RandomAgent('Random')
    pPeaceful = agent.PeacefulAgent('Peaceful')
    
    
    # PUCT Player
    # Create a board just to obtain information needed to create PUCT player
    board_orig = Board(world, [pRandom, pPeaceful])
    board_orig.setPreferences(board_params)
    
    
    
        
    
    # Battle here. Create agent first, then set number of matches and play the games    
    results = {}
    for round in range(num_rounds):
    
        # First game        
        p0, p1 = players[0], players[1]
        board = Board(world, [p0, p1])
        board.setPreferences(prefs)
        
        # Play
        for turn in range(max_turns_per_game):
            board.play()
            if board.gameOver: break
        
        # Get results        
        res = {"p0": p0.name, "p1": p1.name, "gameOver": board.gameOver,
              "p0_countries": p0.num_countries, "p0_armies": board.getPlayerArmies(0), "p0_alive": p0.is_alive,
              "p1_countries": p1.num_countries, "p1_armies": board.getPlayerArmies(1), "p1_alive": p1.is_alive}
        results = append_each_field(results, res)

        # Second game
        p0, p1 = players[1], players[0]
        board = Board(world, [p0, p1])
        board.setPreferences(prefs)
        
        # Play
        for turn in range(max_turns_per_game):
            board.play()
            if board.gameOver: break
        
        # Get results        
        res = {"p0": p0.name, "p1": p1.name, "gameOver": board.gameOver,
              "p0_countries": p0.num_countries, "p0_armies": board.getPlayerArmies(0), "p0_alive": p0.is_alive,
              "p1_countries": p1.num_countries, "p1_armies": board.getPlayerArmies(1), "p1_alive": p1.is_alive}
        results = append_each_field(results, res)
        
    # Write csv with the results
    csv = pd.DataFrame(data = results)
    csv.to_csv(f"../support/battles/{battle_name}.csv")
    print("Battle done")
    
    

