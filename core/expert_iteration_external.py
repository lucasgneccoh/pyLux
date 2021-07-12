
import os
import itertools
import numpy as np
import copy
import sys
import json
import time
import misc

"""
This file is the main file used to perform the Expert iteration process.
It will use system calls to run each step of the process
"""

def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--inputs", help="Path to the json file containing the inputs to the script", default = "../support/exp_iter_inputs/exp_iter_inputs.json")
  parser.add_argument("--verbose", help="Print on the console?", type=int, default = 0)
  args = parser.parse_args()
  return args 



if __name__ == '__main__':
    # ---------------- Start -------------------------
    print("Parsing args")
    args = parseInputs()
    inputs = read_json(args.inputs)
    verbose = bool(args.verbose)
    parallel = bool(args.parallel)
    
    print("ARGS:")
    print(f"verbose: {verbose}")
    print(f"parallel: {parallel}")
    


    iterations = inputs["iterations"]
    num_samples = inputs["num_samples"]
    num_cpu = inputs["num_cpu"]
    saved_states_per_episode = inputs["saved_states_per_episode"]
    max_depth = inputs["max_depth"]
    initial_apprentice_mcts_sims = inputs["initial_apprentice_mcts_sims"]
    expert_mcts_sims = inputs["expert_mcts_sims"]

    params_path = inputs["params_path"]
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
    types_cycle = itertools.cycle(move_types)
                                    
                                    
                                    
    print("Defining apprentice")
    # Define initial apprentice
    apprentice = agent.MctsApprentice(num_MCTS_sims = initial_apprentice_mcts_sims, temp=1, max_depth=max_depth)
    apprentice = agent.NetApprentice(net) # Test the net apprentice, it is way faster # CAMBIAR


    print("Defining expert")
    # build expert
    expert = build_expert_mcts(None) # Start with only MCTS with no inner apprentice
    
    expert = build_expert_mcts(agent.NetApprentice(net)) # Test the network # CAMBIAR
    
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
       
        state = copy.deepcopy(board_orig)
        state.initialPhase = True
        state.pickInitialCountries = True
        
        
        # Samples from self play
        # Use expert to calculate targets
        """
        create_self_play_data(path_data, state, num_samples, num_samples*i,
                              apprentice, expert, max_depth = max_depth,
                              saved_states_per_episode = saved_states_per_episode, verbose=True)
        """
        
        print("Parallel self-play")
        start = time.perf_counter()
        
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
                
        f = lambda t: create_self_play_data(t, path_data, state, apprentice, max_depth = max_depth, saved_states_per_episode=saved_states_per_episode, verbose = verbose)
        # num_samples = iterations * num_cpu * saved_states_per_episode
        num_iter = max (num_samples // (num_cpu * saved_states_per_episode), 1)
        states_to_save = []
        
        for j in range(num_iter):
            types=[]
            for _ in range(num_cpu):
                types.append(next(types_cycle))
            print(f"\t\tIter {j+1} of {num_iter}. Number of processes: {num_cpu}")
            
            # Parallel
            if parallel:
                aux = parmap(f, types, nprocs=num_cpu) 
                for a in aux:
                    for s in a[1]:
                        states_to_save.append(s) # parmap returns this [(i, x)]
            else:
            
                # Sequential (parallel is only working for the first iteration)
                for t in types:
                    if verbose: print(f"\t\tSequential self play: {t}")
                    aux = create_self_play_data(t, path_data, state, apprentice, max_depth = max_depth,
                                          saved_states_per_episode= saved_states_per_episode, verbose = verbose)
                    
                    states_to_save.extend(aux)
        
        print(f"Time taken: {round(time.perf_counter() - start,2)}")
        
        # Tag the states   
        # for s in states_to_save:
        #     print("*** ", s.gamePhase)
          
        print(f"\tTag the states ({len(states_to_save)} states to tag)")  
        start = time.perf_counter()
        # Parallel
        if parallel:
            f = lambda state: tag_with_expert_move(state, expert, verbose=verbose)
            aux = parmap(f, states_to_save, nprocs=num_cpu)
            tagged = [a[1] for a in aux]        

        else:
            # Sequential (parallel is only working for the first iteration)
            tagged = []
            for st in states_to_save:
                t = tag_with_expert_move(st, expert, verbose=verbose)
                tagged.append(t)
          
        print(f"Time taken: {round(time.perf_counter() - start,2)}")
        
        
        
        
        # Save the states
        print("\tSave the states")
        start = time.perf_counter()
        # simple_save_state(path, name, state, policy, value)
        
        # Get the initial number for each move type
        aux_dict = dict(zip(move_types, [len(os.listdir(os.path.join(path_data, t, 'raw')))+1 for t in move_types]))
        
        # Create the input for the parmap including the names of the files, the states and the targets        
        for k in range(len(tagged)):
            state, pol, val = tagged[k]
            phase = state.gamePhase
            new = ('board_{}.json'.format(aux_dict[phase]), phase, state, pol, val)
            tagged[k] = new
            aux_dict[phase] += 1
        
        f = lambda tupl:  simple_save_state(os.path.join(path_data, tupl[1], 'raw'), tupl[0], tupl[2], tupl[3], tupl[4], verbose=verbose)
        aux = parmap(f, tagged, nprocs=num_cpu)   

        print(f"Time taken: {round(time.perf_counter() - start,2)}")
        
    
    
        # Train network on dataset
        print("Training network")
        shuffle(move_types)
        for j, move_type in enumerate(move_types):
            print(f"\t{j}:  {move_type}")
            save_path = f"{path_model}/model_{i}_{j}_{move_type}.tar"
            root_path = f'{path_data}/{move_type}'
            
            if len(os.listdir(os.path.join(root_path, 'raw')))<batch_size: continue
            
            risk_dataset = RiskDataset(root = root_path)
            # TODO: add validation data
            loader = G_DataLoader(risk_dataset, batch_size=batch_size, shuffle = True)
            if verbose: print(f"\tTrain on {root_path}, model = {save_path}")
            train_model(net, optimizer, scheduler, criterion, device,
                        epochs = 10, train_loader = loader, val_loader = None, eval_every = 3,
                        load_path = None, save_path = save_path)


        print(f"Time taken: {round(time.perf_counter() - start,2)}")
        
        print("Building expert")
        # build expert with trained net
        apprentice = agent.NetApprentice(net)
        expert = build_expert_mcts(apprentice)
        expert.num_MCTS_sims = expert_mcts_sims
