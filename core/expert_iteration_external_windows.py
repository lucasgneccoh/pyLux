#%%% Create self play

from board import Board
from world import World
import agent
from model import boardToData, GCN_risk, saveBoardObs, load_dict
import misc

import os
import numpy as np
import copy
import sys

import torch

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import time

# Train model

from model import RiskDataset, train_model, TPT_Loss
import itertools

from torch_geometric.data import DataLoader as G_DataLoader
from random import shuffle

# Expert iteration main

import shutil

#%%% Self play
       
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
        print("\t\tSelf-play starting")
        sys.stdout.flush()
        
    try:
        # Define here how many states to select, and how
        # edge_index = boardToData(root).edge_index    

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
    start = time.process_time()
    _, _, value_exp, Q_value_exp = expert.getBestAction(state, player = state.activePlayer.code, num_sims = None, verbose=verbose)
    policy_exp = expert.getVisitCount(state, temp=temp)
    # TODO: Katago improvement
    
    if isinstance(policy_exp, torch.Tensor):
        policy_exp = policy_exp.detach().numpy()
    if isinstance(value_exp, torch.Tensor):
        value_exp = value_exp.detach().numpy()
    
    if verbose: 
        print(f"\t\tTag with expert: Tagged board {state.board_id} ({state.gamePhase}). {round(time.process_time() - start,2)} sec")
        sys.stdout.flush()
    
    return state, policy_exp, value_exp
    

def simple_save_state(root_path, state, policy, value, verbose=False, num_task=0):
    try:
        board, _ = state.toCanonical(state.activePlayer.code)
        phase = board.gamePhase
        full_path = os.path.join(root_path, phase, 'raw')
        num = len(os.listdir(full_path))+1        
        name = f"board_{num}.json"
        while os.path.exists(os.path.join(full_path, name)):
            num += 1
            name = f"board_{num}.json"
        name = f"board_{num}_{num_task}.json" # Always different
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

def create_self_play_script(input_file, move_type, verbose):
    # ---------------- Start -------------------------
    
    inputs = misc.read_json(input_file)    
    
    misc.print_and_flush("create_self_play: Start")
    start = time.process_time()
    
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

    # Create board
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
        if verbose: misc.print_and_flush("create_self_play: Creating model")
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
        
                if verbose: misc.print_and_flush(f"create_self_play : Chosen model is {model_name}")
                state_dict = load_dict(os.path.join(path_model, model_name), device = 'cpu', encoding = 'latin1')                
                net.load_state_dict(state_dict['model'])        
                if verbose: misc.print_and_flush("create_self_play: Model has been loaded")
            except Exception as e:
                print(e)
                
                                    
        if verbose: misc.print_and_flush("create_self_play: Defining net apprentice")
        # Define initial apprentice        
        apprentice = agent.NetApprentice(net)
    else:
        if verbose: misc.print_and_flush("create_self_play: Defining MCTS apprentice")
        apprentice = agent.MctsApprentice(num_MCTS_sims = apprentice_params["num_MCTS_sims"],
                                          temp = apprentice_params["temp"], 
                                          max_depth = apprentice_params["max_depth"],
                                          sims_per_eval = apprentice_params["sims_per_eval"])


    if verbose: misc.print_and_flush("create_self_play: Defining expert")
    # build expert
    expert = build_expert_mcts(apprentice, max_depth=expert_params["max_depth"],
                    sims_per_eval=expert_params["sims_per_eval"], num_MCTS_sims=expert_params["num_MCTS_sims"],
                    wa = expert_params["wa"], wb = expert_params["wb"],
                    cb = expert_params["cb"], use_val = expert_params["use_val"])
    
                         
    if verbose: misc.print_and_flush("create_self_play: Creating data folders")
    # Create folders to store data
    for folder in move_types:
        os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
    os.makedirs(path_model, exist_ok = True)
                                    

    #### START
    start_inner = time.process_time()
    
    state = copy.deepcopy(board_orig)    

    # Play episode, select states to save
    if verbose: misc.print_and_flush("create_self_play: Self-play")
    
    states_to_save = create_self_play_data(move_type, path_data, state, apprentice, max_depth = max_episode_depth, saved_states_per_episode=saved_states_per_episode, verbose = verbose)

    if verbose: misc.print_and_flush(f"create_self_play: Play episode: Time taken: {round(time.process_time() - start_inner,2)}")
    
    
    # Tag the states and save them
    start_inner = time.process_time()
    if verbose: misc.print_and_flush(f"create_self_play: Tag the states ({len(states_to_save)} states to tag)")  
    for st in states_to_save:
        st_tagged, policy_exp, value_exp = tag_with_expert_move(st, expert, temp=expert_params["temp"], verbose=verbose)
        _ = simple_save_state(path_data, st_tagged, policy_exp, value_exp, verbose=verbose)
    if verbose: misc.print_and_flush(f"create_self_play: Tag and save: Time taken -> {round(time.process_time() - start_inner,2)}")
    
        
    misc.print_and_flush(f"create_self_play: Total time taken -> {round(time.process_time() - start,2)}")
    

#%%% Train model

def train_model_main(input_file, iteration, checkpoint, verbose):
    # ---------------- Start -------------------------
    misc.print_and_flush(f"train_model {iteration}: Start")
    start = time.process_time()
    
    inputs = misc.read_json(input_file)
        
    path_data = inputs["path_data"]
    path_model = inputs["path_model"]
    batch_size = inputs["batch_size"]
    model_args =  misc.read_json(inputs["model_parameters"])

    board_params = inputs["board_params"]
    path_board = board_params["path_board"]    
    
    
    epochs = inputs["epochs"]
    eval_every = inputs["eval_every"]
     

    # ---------------- Load model -------------------------
    
    move_types = ['initialPick', 'initialFortify', 'startTurn', 'attack', 'fortify']

    # Create Board
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

    if verbose: misc.print_and_flush("Creating model")
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
    if verbose: misc.print_and_flush("Training network")
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

    misc.print_and_flush(f"train_model: Total time taken -> {round(time.process_time() - start,2)}")


#%%% Expert Iteration external


def get_last_model(path_model):
  best_i, best_j = -99, -99
  last_model = ""
  for name in os.listdir(path_model):
      try:
          split = name.split("_")
          i, j = int(split[1]), int(split[2])
          if i>best_i or (i==best_i and j>best_j):
              last_model = name
              best_i, best_j = i, j
      except Exception:
          pass
  return last_model
       

def parseInputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputs", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/exp_iter_inputs.json")
    parser.add_argument("--verbose", help="Print on the console?", type=int, default = 0)
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
  
  
    # ---------------- Start -------------------------
    print("Parsing args...", sep = " ")
    args = parseInputs()
    
    #### Manual ####
    args.inputs = "../support/exp_iter_inputs/exp_iter_inputs_small.json"
    args.verbose = 1
    
    
    inputs = misc.read_json(args.inputs)
    verbose = args.verbose

    # Get the parameters for the Expert Iteration process
    iterations = inputs["iterations"]
    num_samples = inputs["num_samples"]
    num_cpu = inputs["num_cpu"]
    saved_states_per_episode = inputs["saved_states_per_episode"]
    max_episode_depth = inputs["max_episode_depth"]
    
    apprentice_params = inputs["apprentice_params"]
    expert_params = inputs["expert_params"]

    params_path = inputs["params_path"]
    path_data = inputs["path_data"]
    path_model = inputs["path_model"]
    batch_size = inputs["batch_size"]
    model_args =  misc.read_json(inputs["model_parameters"])

    board_params = inputs["board_params"]
    path_board = board_params["path_board"]
    
    self_play_tag = inputs["self_play_tag"]
    train_apprentice_tag = inputs["train_apprentice_tag"]
    
    python_command = inputs["python_command"]
    max_seconds_process = inputs["max_seconds_process"]
    
    epochs = inputs["epochs"]
    eval_every = inputs["eval_every"]
    
    print(f"Iterations: {iterations}")
    print(f"Path data: {path_data}")
    print(f"Path model: {path_model}")
    print("Done parsing, starting Expert iteration")
    
    
    move_types = ["initialPick", "initialFortify", "startTurn", "attack", "fortify"]
                         
    print("Creating data folders", sep = " ")
    # Create folders to store data
    for folder in move_types:
        os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
    os.makedirs(path_model, exist_ok = True)
                                    
    print("done")
    
    
    
    
    num_cpu = 1 # Default in local
    
    num_iter = int(num_samples / (num_cpu * saved_states_per_episode))
    
    types_cycle = itertools.cycle(["initialPick", "initialFortify", "startTurn", "attack", "fortify"])
    
    for i in range(iterations):
    
        print(f"\n********* Starting iteration {i+1} *********") 
        
        ##### 1. Self play
        start = time.process_time()
        
        print("Sequential self-play and tagging")
                
        # If data is not kept, erase folders and create new data
        if inputs["delete_previous"]:
            for folder in move_types:
                shutil.rmtree(os.path.join(path_data, folder))
                os.makedirs(os.path.join(path_data, folder, 'raw'))
        
        # Create json file with inputs for the self play tasks
        input_dict = {
          "saved_states_per_episode": saved_states_per_episode,          
          "apprentice_params": apprentice_params,          
          "expert_params": expert_params,          
          "path_data": path_data,
          "path_model": path_model,
          "model_parameters": inputs["model_parameters"],
          "board_params": board_params,
          "max_episode_depth": inputs["max_episode_depth"]          
        }
        self_play_input_json = os.path.join(params_path, self_play_tag) + ".json"
        misc.write_json(input_dict, self_play_input_json)

        misc.print_and_flush(f"Running {num_iter} iterations, each of {num_cpu} tasks")
        for j in range(num_iter):
            # Each iteration launches num_cpu tasks
            misc.print_and_flush(f"\n\t*** Inner iter {j+1} of {num_iter}")
            move_type = next(types_cycle)
            create_self_play_script(self_play_input_json, move_type, verbose)
                        
        print(f"Time taken self-play: {round(time.process_time() - start,2)}")
    
        ##### 2. Train network on dataset
        start = time.process_time()
        print("Training network")
        
        # Create the input file
        input_dict = {
          "path_data": path_data,
          "path_model": path_model,
          "batch_size": batch_size, 
          "model_parameters": inputs["model_parameters"],
          "board_params": board_params,
          "epochs": epochs,
          "eval_every": eval_every
        }
        train_input_json = os.path.join(params_path, train_apprentice_tag) + ".json"
        misc.write_json(input_dict, train_input_json)
        
        checkpoint = apprentice_params["model_name"]
        if not checkpoint:
            checkpoint = get_last_model(path_model)
            
        train_model_main(train_input_json, i, checkpoint, verbose)
        
        
        print(f"Time taken training: {round(time.process_time() - start,2)}")
        
        ##### 3. Update paths so that new apprentice and expert are used on the next iteration
        
        # Change apprentice so that next time the apprentice is the newest neural network
        last_model = get_last_model(path_model)
        apprentice_params = {"type": "net",
                             "model_name": last_model}
        
        # Expert does not change appart from the inner apprentice
        
    