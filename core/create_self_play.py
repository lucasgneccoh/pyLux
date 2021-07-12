from board import Board
from world import World, Country, Continent
from move import Move
import agent
from model import boardToData, GCN_risk, RiskDataset, saveBoardObs, train_model
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

import time

def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--inputs", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/create_self_play.json")
  parser.add_argument("--move_type", help="Type of move to consider", default = "all")
  parser.add_argument("--verbose", help="Print on the console?", default = 0)
  parser.add_argument("--num_task", help="Number of the task for debugging and tracking", default = 0)
  args = parser.parse_args()
  return args
       
def play_episode(root, max_depth, apprentice, move_type = "all", verbose=False):
    episode = []
    state = copy.deepcopy(root)
    edge_index = boardToData(root).edge_index
    # ******************* PLAY EPISODE ***************************
    for i in range(max_depth):  
        #print_message_over(f"Playing episode: {i}/{max_depth}")

        # Check if episode is over            
        if state.gameOver: break

        # Check is current player is alive or not
        if not state.activePlayer.is_alive: 
            # print("\npassing, dead player")
            state.endTurn()
            continue

        # Get possible moves, and apprentice policy
        mask, actions = agent.maskAndMoves(state, state.gamePhase, edge_index)
        
        try:
            policy, value = apprentice.getPolicy(state)
        except Exception as e:
            state.report()
            print(state.activePlayer.is_alive)
            print(state.activePlayer.num_countries)
            raise e
        
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().numpy()
        
        probs = policy * mask             
        
        probs = probs.flatten()
          
        probs =  probs / probs.sum()

        # Random selection? e-greedy?
        
        ind = np.random.choice(range(len(actions)), p = probs)
        move = agent.buildMove(state, actions[ind])
        
        saved = (move_type=="all" or move_type==state.gamePhase)
        if verbose:             
            # print(f"\t\tPlay episode: turn {i}, move = {move}, saved = {saved}")
            pass
        
        if saved:
            episode.append(copy.deepcopy(state))
            
        if move_type == "initialPick":
            if state.gamePhase != "initialPick":
                break
        elif move_type == "initialFortify":
            if state.gamePhase in ["startTurn", "attack", "fortify"]:
                break
                
            
        # Play the move to continue
        state.playMove(move)
        
    return episode
     
def create_self_play_data(move_type, path, root, apprentice, max_depth = 100, saved_states_per_episode=1, verbose = False):
    """ Function to create episodes from self play.
        Visited states are saved and then re visited with the expert to label the data        
    """

    if verbose: 
        print(f"\t\tSelf-play starting")
        sys.stdout.flush()
        
    try:
        # Define here how many states to select, and how
        edge_index = boardToData(root).edge_index    

        # ******************* PLAY EPISODE ***************************
        episode = play_episode(root, max_depth, apprentice, move_type = move_type, verbose = verbose)
        
        # ******************* SELECT STATES ***************************
        # Take some states from episode    
        
        options = [s for s in episode if s.gamePhase == move_type]
        if not options:
            # TODO: What to do in this case? For now just take some random states to avoid wasting the episode
            options = episode
        states_to_save = np.random.choice(options, min(saved_states_per_episode, len(options)), replace=False)
    except Exception as e:
        raise e
    
    if verbose: 
        print(f"\t\tSelf-play done: move_type = {move_type}, {len(states_to_save)} states to save")
        sys.stdout.flush()
        
    return states_to_save


def tag_with_expert_move(state, expert, temp=1, verbose=False):
    # Tag one state with the expert move    
    start = time.perf_counter()
    _, _, value_exp, Q_value_exp = expert.getBestAction(state, player = state.activePlayer.code, num_sims = None, verbose=False)
    policy_exp = expert.getVisitCount(state, temp=temp)
    
    if isinstance(policy_exp, torch.Tensor):
        policy_exp = policy_exp.detach().numpy()
    if isinstance(value_exp, torch.Tensor):
        value_exp = value_exp.detach().numpy()
    
    if verbose: 
        print(f"\t\tTag with expert: Tagged board {state.board_id} ({state.gamePhase}). {round(time.perf_counter() - start,2)} sec")
        sys.stdout.flush()
    
    return state, policy_exp, value_exp
    

def simple_save_state(root_path, state, policy, value, verbose=False):
    try:
        board, _ = state.toCanonical(state.activePlayer.code)
        phase = board.gamePhase
        full_path = os.path.join(root_path, phase, 'raw')
        num = len(os.listdir(full_path))+1        
        name = f"board_{num}.json"
        while os.path.exists(os.path.join(full_path, name)):
            num += 1
            name = f"board_{num}.json"
        saveBoardObs(full_path, name,
                            board, board.gamePhase, policy.ravel().tolist(), value.ravel().tolist())
        if verbose: 
            print(f"\t\tSimple save: Saved board {state.board_id} {os.path.join(full_path, name)}")
            sys.stdout.flush()
        return True
    except Exception as e:
        print(e)
        raise e

def build_expert_mcts(apprentice, max_depth=200, sims_per_eval=1, num_MCTS_sims=1000,
                      wa = 10, wb = 10, cb = np.sqrt(2), use_val = 0):
    return agent.PUCT(apprentice, max_depth = max_depth, sims_per_eval = sims_per_eval, num_MCTS_sims = num_MCTS_sims,
                 wa = wa, wb = wb, cb = cb, use_val = use_val, console_debug = False)

if __name__ == '__main__':
    # ---------------- Start -------------------------
    start = time.perf_counter()
    
    args = parseInputs()
    inputs = misc.read_json(args.inputs)
    move_type = args.move_type
    verbose = args.verbose
    num_task = args.num_task
    
    
    saved_states_per_episode = inputs["saved_states_per_episode"]
    max_episode_depth = inputs["max_episode_depth"]
    apprentice_params = inputs["apprentice_params"]
    expert_params = inputs["expert_params"]
    

    path_data = inputs["path_data"]
    path_model = inputs["path_model"]
    model_args =  misc.read_json(inputs["model_parameters"])

    board_params = inputs["board_params"]
    path_board = board_params["path_board"]
    
    move_types = ["initialPick", "initialFortify", "startTurn", "attack", "fortify"]
    # ---------------------------------------------------------------

    if verbose: misc.print_and_flush("create_self_play: Creating board")

    #%%% Create Board
    world = World(path_board)


    # Set players
    pR1, pR2 = agent.RandomAgent('Red'), agent.RandomAgent('Blue')
    players = [pR1, pR2]
    # Set board
    
    prefs = board_params
            
    board_orig = Board(world, players)
    board_orig.setPreferences(prefs)

    num_nodes = board_orig.world.map_graph.number_of_nodes()
    num_edges = board_orig.world.map_graph.number_of_edges()

    if apprentice_params["type"] == "net":
        if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Creating model")
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
        
        model_name = apprentice_params["model_name"]
        if model_name: # If it is not the empty string
            try:
        
                if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Chosen model is {model_name}")
                state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')                
                net.load_state_dict(state_dict['model'])        
                if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Model has been loaded")
            except Exception as e:
                print(e)
                
                                    
        if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Defining net apprentice")
        # Define initial apprentice        
        apprentice = agent.NetApprentice(net)
    else:
        if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Defining MCTS apprentice")
        apprentice = agent.MctsApprentice(num_MCTS_sims = apprentice_params["num_MCTS_sims"],
                                          temp = apprentice_params["temp"], 
                                          max_depth = apprentice_params["max_depth"],
                                          sims_per_eval = apprentice_params["sims_per_eval"])


    if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Defining expert")
    # build expert
    expert = build_expert_mcts(apprentice, max_depth=expert_params["max_depth"],
                    sims_per_eval=expert_params["sims_per_eval"], num_MCTS_sims=expert_params["num_MCTS_sims"],
                    wa = expert_params["wa"], wb = expert_params["wb"],
                    cb = expert_params["cb"], use_val = expert_params["use_val"])
    
                         
    if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Creating data folders")
    # Create folders to store data
    for folder in move_types:
        os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
    os.makedirs(path_model, exist_ok = True)
                                    

    #### START
    start_inner = time.perf_counter()
    
    state = copy.deepcopy(board_orig)    

    # Play episode, select states to save
    if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Self-play")
    
    states_to_save = create_self_play_data(move_type, path_data, state, apprentice, max_depth = max_episode_depth, saved_states_per_episode=saved_states_per_episode, verbose = verbose)

    if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Play episode: Time taken: {round(time.perf_counter() - start_inner,2)}")
    
    
    # Tag the states and save them
    start_inner = time.perf_counter()
    if verbose: misc.print_and_flush(f"create_self_play ({num_task}): Tag the states ({len(states_to_save)} states to tag)")      
    for st in states_to_save:
        st_tagged, policy_exp, value_exp = tag_with_expert_move(st, expert, temp=expert_params["temp"], verbose=verbose)
        res = simple_save_state(path_data, st_tagged, policy_exp, value_exp, verbose=verbose)
    if verbose: misc.print_and_flush(f"create_self_play  ({num_task}): Tag and save: Time taken -> {round(time.perf_counter() - start_inner,2)}")
    
        
    if verbose: misc.print_and_flush(f"create_self_play  ({num_task}): Total time taken -> {round(time.perf_counter() - start,2)}")
    
