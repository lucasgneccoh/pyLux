# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:08:40 2021

@author: lucas
"""

# Create a model (random weighted neural network) so that the first iteration 
# of Exp It does not need the MCTS apprentice which takes a lot of time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import misc
from model import save_dict, GCN_risk
from board import Board, World, RandomAgent
import torch

def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--inputs", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/exp_iter_inputs.json")
  parser.add_argument("--save_path", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/model_random.tar")
  args = parser.parse_args()
  return args 


if __name__ == "__main__":
    
    args = parseInputs()  
    inputs = misc.read_json(args.inputs)
    save_path = args.save_path
    
    model_args =  misc.read_json(inputs["model_parameters"])
    board_params = inputs["board_params"]    
    path_board = board_params["path_board"]
    
    
    print(f"Creating board for map {path_board}")
        
        
    world = World(path_board)


    # Set players
    pR1, pR2 = RandomAgent('Red'), RandomAgent('Blue')
    players = [pR1, pR2]
    # Set board
    prefs = board_params
            
    board_orig = Board(world, players)
    board_orig.setPreferences(prefs)

    num_nodes = board_orig.world.map_graph.number_of_nodes()
    num_edges = board_orig.world.map_graph.number_of_edges()

    print(f"Creating model with parameters from {inputs['model_parameters']}")
    net = GCN_risk(num_nodes, num_edges, 
                     model_args['board_input_dim'], model_args['global_input_dim'],
                     model_args['hidden_global_dim'], model_args['num_global_layers'],
                     model_args['hidden_conv_dim'], model_args['num_conv_layers'],
                     model_args['hidden_pick_dim'], model_args['num_pick_layers'], model_args['out_pick_dim'],
                     model_args['hidden_place_dim'], model_args['num_place_layers'], model_args['out_place_dim'],
                     model_args['hidden_attack_dim'], model_args['num_attack_layers'], model_args['out_attack_dim'],
                     model_args['hidden_fortify_dim'], model_args['num_fortify_layers'], model_args['out_fortify_dim'],
                     model_args['hidden_value_dim'], model_args['num_value_layers'],
                     model_args['dropout'], model_args['block'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    
    print("Net ready, creating other elements of the .tar file")
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.97)
    
    save_dict(save_path, {'model':net.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'epoch': 0, 'best_loss':99999})
    
    print(f"Model saved in {save_path}")