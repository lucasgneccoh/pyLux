
import os
import itertools
import numpy as np
import copy
import sys
import json
import time
import subprocess
from subprocess import Popen
from threading import Timer
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
      except Exception as e:
          pass
  return last_model
          

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
    max_seconds_process = inputs["max_seconds_process"]
    
    epochs = inputs["epochs"]
    eval_every = inputs["eval_every"]
    
    print("done")
    
    
    move_types = ["initialPick", "initialFortify", "startTurn", "attack", "fortify"]
                         
    print("Creating data folders", sep = " ")
    # Create folders to store data
    for folder in move_types:
        os.makedirs(os.path.join(path_data, folder, 'raw'), exist_ok = True)
    os.makedirs(path_model, exist_ok = True)
                                    
    print("done")
    
    
    
    # Calculate number of tasks of self-play
    # Here, we launch num_cpu tasks, so it all comes down to calculate the number of iterations
    # num_samples = num_iter * num_cpu * saved_states_per_episode
    # --> num_iter = int(num_samples / (num_cpu * saved_states_per_episode))
    num_iter = int(num_samples / (num_cpu * saved_states_per_episode))
    
    types_cycle = itertools.cycle(["initialPick", "initialFortify", "startTurn", "attack", "fortify"])
    
    for i in range(iterations):
    
        print(f"\n********* Starting iteration {i+1} *********") 
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
        misc.write_json(input_dict, self_play_input_json)

        misc.print_and_flush(f"Running {num_iter} iterations, each of {num_cpu} tasks")
        for j in range(num_iter):
            # Each iteration launches num_cpu tasks
            misc.print_and_flush(f"\tInner iter {j+1} of {num_iter}")
            processes = []
            
            for k in range(num_cpu):
                move_type = next(types_cycle)
                #subprocess.run(["taskset", "-c", str(k), python_command, f"{self_play_tag}.py", "--inputs", self_play_input_json, "--move_type", move_type, "--verbose", str(verbose), "--num_task", str(k)])
                processes.append((k, Popen(["taskset", "-c", str(k), python_command, f"{self_play_tag}.py", "--inputs",             self_play_input_json, "--move_type", move_type, "--verbose", str(verbose), "--num_task", str(k)])))
            
            timer_start = time.perf_counter()
            while processes:                
                for k, p in processes:
                    if p.poll() is not None: # process ended
                        print(f"Process {k} finished") 
                        processes.remove((k, p))
                # Check timer
                if time.perf_counter() - timer_start > max_seconds_process:
                    print(f"*** TIMEOUT: Killing all remaining processes ({len(processes)})")
                    for k, p in processes: 
                        p.kill()
                    processes = []
                        
              
          
        print(f"Time taken: {round(time.perf_counter() - start,2)}")
    
        ##### 2. Train network on dataset
        start = time.perf_counter()
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
            get_last_model(path_model)
        subprocess.run([python_command, f"{train_apprentice_tag}.py", "--inputs", train_input_json, "--iteration", str(i), "--verbose", str(verbose), "--checkpoint", get_last_model(path_model)])
        
        
        print(f"Time taken: {round(time.perf_counter() - start,2)}")
        
        ##### 3. Update paths so that new apprentice and expert are used on the next iteration