from board import Board
from world import World
import agent
from model import GCN_risk, load_dict
import misc


import torch


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
    model_args =  misc.read_json(args["model_parameters_json"])
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
    net.eval()
    
    state_dict = load_dict(args["model_path"], device = 'cpu', encoding = 'latin1')

    net.load_state_dict(state_dict['model'])   

    apprentice = agent.NetApprentice(net)
             
    kwargs = {}
    for a in ["sims_per_eval", "num_MCTS_sims", "wa", "wb", "cb", "temp", "use_val", "verbose"]:
        if a in args: kwargs[a] = args[a]
    pPUCT = agent.PUCTPlayer(apprentice = apprentice, **kwargs)

    
    return pPUCT
    
def load_NetPlayer(board, args):
    num_nodes = board.world.map_graph.number_of_nodes()
    num_edges = board.world.map_graph.number_of_edges()    
    model_args =  misc.read_json(args["model_parameters_json"])
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
    net.eval()
    
    state_dict = load_dict(args["model_path"], device = 'cpu', encoding = 'latin1')

    net.load_state_dict(state_dict['model'])   

    apprentice = agent.NetApprentice(net)
    
    kwargs = {}
    for a in ["move_selection", "name", "temp"]:
        if a in args: kwargs[a] = args[a]
    netPlayer = agent.NetPlayer(apprentice, **kwargs)
    
    return netPlayer
    
def create_player_list(args):
    # Only need board_params and players in args
    board_params = args["board_params"]    

    list_players = []
    for i, player_args in enumerate(args["players"]):
        kwargs = removekey(player_args, "agent")
        if player_args["agent"] == "RandomAgent":
            list_players.append(agent.RandomAgent(f"Random_{i}"))
        elif player_args["agent"] == "PeacefulAgent":
            list_players.append(agent.PeacefulAgent(f"Peaceful_{i}"))
        elif player_args["agent"] == "FlatMCPlayer":
            list_players.append(agent.FlatMCPlayer(name=f'flatMC_{i}', **kwargs))
        elif player_args["agent"] == "UCTPlayer":
            list_players.append(agent.UCTPlayer(name=f'UCT_{i}', **kwargs))
        elif player_args["agent"] == "PUCTPlayer":            
            world = World(board_params["path_board"])
            board = Board(world, [agent.RandomAgent('Random1'), agent.RandomAgent('Random2')])
            board.setPreferences(board_params)
            puct = load_puct(board, player_args)
            list_players.append(puct)
        elif player_args["agent"] == "NetPlayer":            
            world = World(board_params["path_board"])
            board = Board(world, [agent.RandomAgent('Random1'), agent.RandomAgent('Random2')])
            board.setPreferences(board_params)
            netPlayer = load_NetPlayer(board, player_args)
            list_players.append(netPlayer)
        elif player_args["agent"] == "Human":
            hp_name = player_args["name"] if "name" in player_args else "human"
            hp = agent.HumanAgent(name=hp_name)
            list_players.append(hp)
            
    return list_players
  
def battle(args):
    results = {}
    
    # Create players and board    
    board_params = args["board_params"]
    M = args["max_turns_per_game"]
    for i in range(args["num_rounds"]):
        world = World(board_params["path_board"])
        list_players = create_player_list(args)
        board = Board(world, list_players)
        board.setPreferences(board_params)
        
        print(f"\t\tRound {i+1}")
        for j in range(M):
            aux = "{: <30}".format(f"Player {board.activePlayer.name} playing")
            misc.print_message_over(f"\t\t\t{aux} Turn {j:0>4}")
            board.play()            
            if board.gameOver: break
        print()
        winner = "Nobody"
        if board.gameOver:
            for _, p in board.players.items():
                if p.is_alive:
                    winner = p.name
                    break
        print(f"\t\tDone: Winner is {winner}")
        append_each_field(results, {"round": i})
        for k in board.players:
            append_each_field(results, player_results(board, k))
            
        world = None
        board = None
            
    return results
  
def player_results(board, player_code):
    p = board.players[player_code]
    c = player_code
    n = p.name
    return {f"{n}_armies": board.getPlayerArmies(c),
            f"{n}_income": board.getPlayerIncome(c),
            f"{n}_countries": board.getPlayerCountries(c),
            f"{n}_continents": board.getPlayerContinents(c)}

if __name__ == '__main__':
    
    args = parseInputs()
    
    # Manual
    # args.inputs = "../support/battles/diamond_baselines.json"
    
    
    inputs = misc.read_json(args.inputs)

    board_params = inputs["board_params"]
    battles = inputs["battles"]
          
    
    # Battle here. Create agent first, then set number of matches and play the games        
    for b_name, b_args in battles.items():
        print(f"Playing battle {b_name}")
        battle_args = dict(b_args)
        battle_args["board_params"] = dict(board_params)
        res = battle(battle_args)                
        # Write csv with the results
        csv, path = pd.DataFrame(data = res), f"../support/battles/{b_name}.csv"
        csv.to_csv(path)
        print(f"Wrote results to {path}")
        
    
    print("Battle done")
    
    

