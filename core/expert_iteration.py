import agent
from board import Board
from world import World, Country, Continent
from move import Move
from mcts import MCTS, MctsApprentice, NetApprentice, maskAndMoves, buildMove
from model import boardToData, GCN_risk, RiskDataset, saveBoardObs, train_model

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

import multiprocessing
from multiprocessing import Pool, cpu_count

from random import shuffle

import time

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

def build_expert_mcts(apprentice, num_MCTS_sims = 10000, sims_per_eval = 5, max_depth = 50):
    return MCTS(root=None, apprentice=apprentice, max_depth = max_depth, 
                sims_per_eval = sims_per_eval, num_MCTS_sims = num_MCTS_sims,
                wa = 10, wb = 10, cb = np.sqrt(2))

def isint(n):
    try:
        x = int(n)
        return True
    except:
        return False
     

def play_episode(root, max_depth, apprentice):
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
        
    return episode
     
def create_self_play_data(move_type, path, root, apprentice, max_depth = 100, saved_states_per_episode=1, verbose = False):
    """ Function to create episodes from self play.
        Visited states are saved and then re visited with the expert to label the data
        
        To do this in parallel, we use the multiprocessing library. The idea is to feed a queue of states
        Then use this queue to, in parallel, tag each state with the expert move
    """
    
    """
    samples_type = {'initialPick':0, 'initialFortify':0, 'startTurn':0, 'attack':0, 'fortify':0}
    
    # Get information about existing files, to continue enlarging the dataset
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

    """
    
    edge_index = boardToData(root).edge_index    

    # ******************* PLAY EPISODE ***************************
    episode = play_episode(root, max_depth, apprentice)
        
    # ******************* SELECT STATES ***************************
    # Take some states from episode    
    try:
        # Define here how many states to select, and how
        options = [s for s in episode if s.gamePhase == move_type]
        if not options:
            # TODO: What to do in this case? For now just take some random states to avoid wasting the episode
            options = episode
        states_to_save = np.random.choice(options, min(saved_states_per_episode, len(options)))
    except Exception as e:
        raise e
        
    return states_to_save


def tag_with_expert_move(state, expert):
    # Tag one state with the expert move
    # TODO: expert can be done in parallel?
    policy_exp, value_exp, _ = expert.getActionProb(state, temp=1, num_sims = None, use_val = False)
    if isinstance(policy_exp, torch.Tensor):
        policy_exp = policy_exp.detach().numpy()
    if isinstance(value_exp, torch.Tensor):
        value_exp = value_exp.detach().numpy()    
    return state, policy_exp, value_exp
    
def save_states(path, states, policies, values):
    for state, policy_exp, value_exp in zip(states, policies, values):
        # Save the board, value and target
        board, _ = state.toCanonical(state.activePlayer.code)
        phase = board.gamePhase
        full_path = os.path.join(path, phase, 'raw')
        num = len(os.listdir(full_path))+1
        saveBoardObs(full_path, 'board_{}.json'.format(num),
                        board, board.gamePhase, policy_exp.ravel().tolist(), value_exp.ravel().tolist())

def simple_save_state(path, name, state, policy, value):
    board, _ = state.toCanonical(state.activePlayer.code)
    saveBoardObs(path, name,
                        board, board.gamePhase, policy.ravel().tolist(), value.ravel().tolist())
    return True

def whole_process(args):
    path, root = args['path'], args['root']    
    apprentice, expert = args['apprentice'], args['expert']
    max_depth = args['max_depth']
    saved_states_per_episode, verbose = args['saved_states_per_episode'], args['verbose']
    # This is one process. Another function will do it in parallel
    move_type = args['move_type']
    states_to_save = create_self_play_data(move_type, path, root, apprentice, max_depth, saved_states_per_episode, verbose)
    policies, values = [], []
    for state in states_to_save:
        _, policy_exp, value_exp = tag_with_expert_move(state, expert)
        policies.append(policy_exp)
        values.append(value_exp)
    
    save_states(path, states_to_save, policies, values)
    return move_type


    
def par_self_play(num_samples, path, root, apprentice, expert, num_cpu, max_depth = 100, saved_states_per_episode=1, verbose = False):
    # On the second iteration, this throws an Error 24 Too many open files
    cpus = cpu_count()
    args = dict(zip(["path", "root", "apprentice", "expert", "max_depth", "saved_states_per_episode", "verbose"], [path, root, apprentice, expert, max_depth, saved_states_per_episode, verbose]))
    num_proc = min(cpus, num_cpu)
    num_iter = max(num_samples // (num_proc*saved_states_per_episode), 1)
    move_types = itertools.cycle(['initialPick', 'initialFortify', 'startTurn', 'attack', 'fortify'])
    args_list = []
    for a in range(num_proc):
        copy_a = copy.deepcopy(args)
        copy_a['move_type'] = next(move_types)
        args_list.append(copy_a)    
    
    print("\tStarting pool calls")
    for i in range(num_iter):
        print(f"\t\tStarting with iteration {i+1} of {num_iter}")
        with Pool(num_proc) as pool:
            print(pool.map(whole_process, args_list))          
            pool.close()
            pool.terminate()
        print(f"\t\tDone with iteration {i+1} of {num_iter}")
    
    
    
def par_fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=multiprocessing.cpu_count()):
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=par_fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [(i, x) for i, x in res]



def TPT_Loss(output, target):
    # TODO: Add regularisation
    return -(target*torch.log(output)).sum()



if __name__ == '__main__':
    # ---------------- Start -------------------------
    print("Parsing args")
    args = parseInputs()
    inputs = read_json(args.inputs)


    iterations = inputs["iterations"]
    num_samples = inputs["num_samples"]
    num_cpu = inputs["num_cpu"]
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

    print("Creating board")

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



    move_types = ['initialPick', 'initialFortify', 'startTurn', 'attack', 'fortify']
                                    
                                    
                                    
    print("Defining apprentice")
    # Define initial apprentice
    apprentice = MctsApprentice(num_MCTS_sims = initial_apprentice_mcts_sims, temp=1, max_depth=max_depth)
    # apprentice = NetApprentice(net)


    print("Defining expert")
    # build expert
    expert = build_expert_mcts(None) # Start with only MCTS with no inner apprentice
    expert.num_MCTS_sims = expert_mcts_sims
                         
    print("Creating data folders")
    # Create folders to store data
    for folder in move_types:
        os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
    os.makedirs(path_model, exist_ok = True)
                                    

    state = copy.deepcopy(board_orig)
    state.initialPhase = True
    state.pickInitialCountries = True
    # state.play() # random init
    for i in range(iterations):
        print(f"Starting iteration {i+1}")
       
        # Samples from self play
        # Use expert to calculate targets
        """
        create_self_play_data(path_data, state, num_samples, num_samples*i,
                              apprentice, expert, max_depth = max_depth,
                              saved_states_per_episode = saved_states_per_episode, verbose=True)
        """
        
        print("Parallel self-play")
        
        """
        # Method 1
        par_self_play(num_samples, path_data, state, 
                      apprentice, expert, num_cpu, max_depth = max_depth, saved_states_per_episode=saved_states_per_episode,
                      verbose = True)        
        """
        
        """
        # Method 2. do it manually
        """
        
        # Play the games
        print("\tPlay the games")
        types = []        
        for _ in range(num_cpu):
          types.append(next(itertools.cycle(move_types)))
        f = lambda t: create_self_play_data(t, path_data, state, apprentice, max_depth = max_depth, saved_states_per_episode=saved_states_per_episode, verbose = False)        
        # num_samples = iterations * num_cpu * saved_states_per_episode
        num_iter = max (num_samples // (num_cpu * saved_states_per_episode), 1)
        states_to_save = []
        for j in range(num_iter):
            aux = parmap(f, types, nprocs=num_cpu) 
            for a in aux:
                for s in a[1]
                    states_to_save.append(s) # parmap returns this [(i, x)]
        
        print("States to save: ", len(states_to_save))
        print(states_to_save[0])
        
        
        
        # Tag the states    
        print("\tTag the states")
        f = lambda state: tag_with_expert_move(state, expert)
        aux = parmap(f, states_to_save, nprocs=num_cpu)
        tagged = [a[1] for a in aux]
        print("example")
        print(tagged[0])
        
        # Save the states
        print("\tSave the states")
        # simple_save_state(path, name, state, policy, value)
        
        # Get the initial number for each move type
        aux_dict = dict(zip(move_types, [len(os.listdir(os.path.join(path_data, t, 'raw')))+1 for t in move_types]))
        
        # Create the input for the parmap including the names of the files, the states and the targets        
        for k in range(len(tagged)):
            state, pol, val = tagged[k]
            phase = state.gamePhase
            new = ('board_{}.json'.format(aux_dict[phase]), state, pol, val)
            tagged[k] = new
            aux_dict[phase] += 1
        
        f = lambda tupl:  simple_save_state(path_data, tupl[0], tupl[1], tupl[2], tupl[3])
        aux = parmap(f, tagged, nprocs=num_cpu)        
        
        print("Training network")
        
        # Train network on dataset
        shuffle(move_types)
        for j, move_type in enumerate(move_types):
            print(f"\t{j}:  {move_type}")
            save_path = f"{path_model}/model_{i}_{j}_{move_type}.tar"
            root_path = f'{path_data}/{move_type}'
            
            if len(os.listdir(os.path.join(root_path, 'raw')))<batch_size: continue
            
            risk_dataset = RiskDataset(root = root_path)
            # TODO: add validation data
            loader = G_DataLoader(risk_dataset, batch_size=batch_size, shuffle = True)
            train_model(net, optimizer, scheduler, criterion, device,
                        epochs = 10, train_loader = loader, val_loader = None, eval_every = 3,
                        load_path = None, save_path = save_path)

        print("Building expert")
        # build expert with trained net
        expert = build_expert_mcts(NetApprentice(net))
