from board import Board
from world import World, Country, Continent
from move import Move
import agent
from model import boardToData, GCN_risk, RiskDataset, saveBoardObs, train_model, TPT_Loss
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

from random import shuffle

import time

def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--inputs", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/train_apprentice.json")
  parser.add_argument("--verbose", help="Print on the console?", type=int, default = 0)
  parser.add_argument("--checkpoint", help="Checkpoint to use when loading the model, optimizer and scheduler", default = "")
  parser.add_argument("--iteration", help="Global iteration of Expert Iteration", type=int, default = 0)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
    # ---------------- Start -------------------------
    print("Parsing args")
    args = parseInputs()
    inputs = read_json(args.inputs)
    verbose = bool(args.verbose)
    iteration = args.iteration
    checkpoint = args.checkpoint
        
    path_data = inputs["path_data"]
    path_model = inputs["path_model"]
    batch_size = inputs["batch_size"]
    model_args =  read_json(inputs["model_parameters"])

    board_params = inputs["board_params"]
    path_board = board_params["path_board"]    
    
    
    epochs = inputs["epochs"]
    eval_every = inputs["eval_every"]
     

    # ---------------- Load model -------------------------
    
    move_types = ['initialPick', 'initialFortify', 'startTurn', 'attack', 'fortify']
    
    print("Creating board to get board attributes")

    #%%% Create Board
    world = World(path_board)


    # Set players
    pR1, pR2 = agent.RandomAgent('Red'), agent.RandomAgent('Blue')
    players = [pR1, pR2]
    # Set board
    # TODO: Send to inputs
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
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = TPT_Loss
    

    #state_dict = model.load_dict(os.path.join(path_model, checkpoint), device = 'cpu', encoding = 'latin1')
    #net.load_state_dict(state_dict['model'])
    #optimizer.load_state_dict(state_dict['optimizer'])
    #scheduler.load_state_dict(state_dict['scheduler'])
    load_path = os.path.join(path_model, checkpoint) if checkpoint else None 
    # This is used only at the beginning. Then the model that is loaded is trained and saved at each time.
    # We avoid reloading the last saved model
    
        
        
    # Train network on dataset
    print("Training network")
    shuffle(move_types)
    for j, move_type in enumerate(move_types):
        if verbose: misc.print_and_flush(f"\tTraining {j}:  {move_type}")
        save_path = f"{path_model}/model_{iteration}_{j}_{move_type}.tar"
        root_path = f'{path_data}/{move_type}'
        
        if len(os.listdir(os.path.join(root_path, 'raw')))<batch_size: continue
        
        risk_dataset = RiskDataset(root = root_path)
        # TODO: add validation data
        loader = G_DataLoader(risk_dataset, batch_size=batch_size, shuffle = True)
        if verbose: misc.print_and_flush(f"\tTrain on {root_path}, model = {save_path}")
        train_model(net, optimizer, scheduler, criterion, device,
                    epochs = epochs, train_loader = loader, val_loader = None, eval_every = eval_every,
                    load_path = load_path, save_path = save_path)
        
        load_path = None # The model is already in memory

    if verbose: misc.print_and_flush(f"Time taken: {round(time.perf_counter() - start,2)}")
        
