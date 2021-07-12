
import os
import itertools
import numpy as np
import copy
import sys
import json
import time
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

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
    print("Parsing args...", sep = " ")
    args = parseInputs()
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
    
    print("done")
    
    
    move_types = ["initialPick", "initialFortify", "startTurn", "attack", "fortify"]
                         
    print("Creating data folders", sep = " ")
    # Create folders to store data
    for folder in move_types:
        os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
    os.makedirs(path_model, exist_ok = True)
                                    
    print("done")
    
    print("\n\n")
    
    # Calculate number of tasks of self-play
    # Here, we launch num_cpu tasks, so it all comes down to calculate the number of iterations
    # num_samples = num_iter * num_cpu * saved_states_per_episode
    # --> num_iter = int(num_samples / (num_cpu * saved_states_per_episode))
    num_iter = int(num_samples / (num_cpu * saved_states_per_episode))
    
    types_cycle = itertools.cycle(["initialPick", "initialFortify", "startTurn", "attack", "fortify"])
    
    for i in range(iterations):
    
        print(f"Starting iteration {i+1}")                
        ##### 1. Self play
        
        print("Parallel self-play and tagging")
        start = time.perf_counter()
        
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
        misc.write_json(input_dict, os.path.join(params_path, self_play_tag))

        print(f"Running {num_iter} iterations, each of {num_cpu} tasks")
        for j in range(num_iter):
            # Each iteration launches num_cpu tasks
            for k in range(num_cpu):
                move_type = next(types_cycle)
                subprocess.run(["taskset", "-c", str(k), python_command, f"{self_play_tag}.py", "--inputs", self_play_input_json, "--move_type", move_type, "--verbose", str(verbose)])
            
        
        
        # For each task, execute the system call usin taskset
        #   Each task must create the episode, select states to save, tag them and save them
        #     This includes loading the current model
        
          
        print(f"Time taken: {round(time.perf_counter() - start,2)}")
        
        break
    
        ##### 2. Train network on dataset
        
        print("Training network")
        shuffle(move_types)
        for j, move_type in enumerate(move_types):
            print(f"\t{j}:  {move_type}")
            save_path = f"{path_model}/model_{i}_{j}_{move_type}.tar" # TODO: add something to the name
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
        
        ##### 3. Update paths so that new apprentice and expert are used on the next iteration