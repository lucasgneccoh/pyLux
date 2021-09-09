from board import Board # Only used for the Board.fromDicts
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch_geometric 

from torch_geometric.nn import GCNConv
from torch_geometric.data import Dataset as G_Dataset

from torch_geometric import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(path, epoch, model, optimizer, scheduler=None, other={}):
    state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
    }
    if not scheduler is None: state_dict['scheduler'] = scheduler.state_dict()
    state_dict.update(other)
    torch.save(state_dict, path)

def load_checkpoint(load_path, model, optimizer, device):
    try:
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        return state_dict
    except Exception as e:
        print(f"Error while opening {load_path}")
        raise e

def save_dict(save_path, state_dict):
    torch.save(state_dict, save_path)
    
def load_dict(load_path, device, encoding = 'utf-8'):
    state_dict = torch.load(load_path, map_location=device, encoding = encoding)
    return state_dict


### Dataset

def countryToTensor(c, board):    
    cont = board.world.continents[int(c.continent)]
    # TODO: finish the country representation, adding features such as is border of continent
    return torch.FloatTensor(list(map(float,[c.code, c.continent, cont.bonus, c.owner, c.armies, c.moveableArmies])))

def boardToData(board):
    ''' Convert a pyLux board into a torch geometric Data object
    '''
    G = board.world.map_graph
    data = utils.from_networkx(G)
    countries = board.countries()
    data.x = torch.stack([torch.cat([countryToTensor(c, board), torch.tensor(oneHotGamePhase(board.gamePhase))], dim=0) for c in countries], dim=0)
    data.edge_index = torch.LongTensor(sorted(list(G.edges))).t().contiguous().view(2, -1)
    data.num_nodes = len(countries)
    return data

def saveBoardObs(path, file_name, board, phase, target, value):
    temp = os.path.join(path, file_name)
    if os.path.exists(temp):
        print(f"File {temp} found: deleting")
        os.remove(temp)
    else:
        # Create file to have it in dir and avoid other processes not counting it
        with open(temp, 'w') as f:
            pass
      
    with open(temp, 'w') as f:
        continents, countries, inLinks, players, misc = board.toDicts()
        json.dump({'continents':continents, 'countries':countries,
                    'inLinks': inLinks, 'players':players, 'misc':misc,
                    'y': target, 'value':value, 'phase':phase}, f)

def playerGlobalFeatures(p, maxIncome, maxCountries):
    if p is None:
        return [0]*4
    return [int(p['cards'])/6, int(p['income'])/maxIncome, int(p['countries'])/maxCountries, int(p['alive'])]

def oneHotGamePhase(phase):
    res = [0]*6
    phases = ['initialPick', 'initialFortify', 'startTurn', 'attack', 'fortify']
    res[phases.index(phase)] = 1
    return res

def buildGlobalFeature(players, misc):
    res = []
    maxIncome, maxCountries = 1,1
    for c, p in players.items():
        maxIncome = max(maxIncome, int(p['income']))
        maxCountries = max(maxCountries, int(p['countries']))

    num_players = len(players)
    func = int if isinstance(list(players.keys())[0], int) else str
    for i in range(num_players): # Max 6 players
        res.extend(playerGlobalFeatures(players[func(i)], maxIncome, maxCountries))
    res.extend([int(misc['nextCashArmies'])/100])
    res.extend(oneHotGamePhase(misc['gamePhase']))
    return torch.FloatTensor(res)
    
class RiskDataset(G_Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        ''' root is the path to save the information '''        
        self._raw_file_names = None
        self.root = root
        super(RiskDataset, self).__init__(root, transform, pre_transform) 
        

    @property
    def raw_file_names(self):        
        if self._raw_file_names is None:
            self._raw_file_names = os.listdir(self.raw_dir)        
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return [n.replace('raw', 'processed').replace('json', 'pt') for n in self.raw_file_names]
    
    def process(self):        
        for i, raw_path in enumerate(self.raw_paths):
            name = raw_path.replace('raw', 'processed').replace('json', 'pt')
            if not os.path.exists(name):
                # Read data from raw_path.
                try:
                    with open(raw_path, 'r') as f:
                        b = json.load(f)                  
                        board = Board.fromDicts(b['continents'], 
                                                b['countries'], b['inLinks'], 
                                                b['players'], b['misc'])
                                        
                        data = boardToData(board)                
                        global_x = buildGlobalFeature(b['players'], b['misc'])               
                        # For some reason this was not working properly                           
                        data.y = torch.Tensor(b['y'])
                        data.value = torch.Tensor(b['value'])
                        data.phase = b['misc']['gamePhase']       
                        
                        

                    #if (not self.pre_filter is None) and (not self.pre_filter(data)):
                    #    continue

                    #if self.pre_transform is not None:
                    #    data = self.pre_transform(data)

                    torch.save({'data':data,'global':global_x}, name)
                except Exception as e:
                    print(e)
                    print(raw_path)
                    raise e  # For now just avoid the interruption of the program, and continue ?

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):        
        read = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx])) 
        return read['data'], read['global']
        
        
 ### Model
 
def ResGCN(conv, norm, act = nn.ReLU(), dropout = 0.3):
    return torch_geometric.nn.DeepGCNLayer(conv, norm, act, block='res+', dropout=dropout)

class EdgeNet(torch.nn.Module):

    def __init__(self, node_dim, hidden_dim, num_layers, out_dim):    
        super().__init__()    
        self.num_layers = num_layers
        dims = [2*node_dim] + [hidden_dim] * (num_layers-1)  + [out_dim]        
        self.layers = nn.ModuleList([nn.Linear(dims[i],dims[i+1]) for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(dims[i+1]) for i in range(num_layers-1)]) 
    
    def forward(self, v1, v2):
        x = torch.cat([v1,v2], dim =1)
        for i in range(self.num_layers-1):
            # print(x.shape)
            x  = self.layers[i](x)
            # print(x.shape)
            # print("-"*30)
            x = F.relu(x)
            x = self.bns[i](x) 
        
        x  = torch.sigmoid(self.layers[-1](x))

        return x


class GCN_risk(torch.nn.Module):
    def __init__(self, num_nodes, num_edges, board_input_dim, global_input_dim,
                 hidden_global_dim = 32, num_global_layers = 4,
                 hidden_conv_dim = 16, num_conv_layers=4, 
                 hidden_pick_dim = 32, num_pick_layers = 4, out_pick_dim = 1,
                 hidden_place_dim = 32, num_place_layers = 4, out_place_dim = 1,
                 hidden_attack_dim = 32, num_attack_layers = 4, out_attack_dim = 1,
                 hidden_fortify_dim = 32, num_fortify_layers = 4, out_fortify_dim = 1,
                 hidden_value_dim = 32,  num_value_layers = 4,
                 dropout = 0.4):

        super().__init__()
       
        # Global
        
        #self.num_global_layers = num_global_layers
        #self.global_fc = torch.nn.ModuleList([nn.Linear(global_input_dim, hidden_global_dim)] + \
        #                                     [nn.Linear(hidden_global_dim, hidden_global_dim) for i in range(num_global_layers-1)])
        #self.global_bns = nn.ModuleList([torch.nn.BatchNorm1d(hidden_global_dim) for i in range(num_global_layers-1)])
        
        self.num_nodes = num_nodes
        self.num_edges = num_edges


        # Board
        self.num_conv_layers = num_conv_layers
        self.conv_init = GCNConv(board_input_dim, hidden_conv_dim)
        self.deep_convs = nn.ModuleList([ResGCN(GCNConv(hidden_conv_dim,hidden_conv_dim),
                                                nn.BatchNorm1d(hidden_conv_dim)) for i in range(num_conv_layers)])        
        
        self.softmax = nn.LogSoftmax(dim = 1)

        # Pick country head        
        self.num_pick_layers = num_pick_layers
        self.pick_layers = torch.nn.ModuleList([ResGCN(GCNConv(hidden_conv_dim,hidden_pick_dim),
                                                nn.BatchNorm1d(hidden_pick_dim))]+\
                                                [ResGCN(GCNConv(hidden_pick_dim,hidden_pick_dim),
                                                nn.BatchNorm1d(hidden_pick_dim)) for i in range(num_pick_layers-2)] +\
                                               [GCNConv(hidden_pick_dim, out_pick_dim)]
                                               )
        self.pick_final = torch.nn.ModuleList([nn.Linear(num_nodes, 64), nn.Linear(64, num_nodes)])

        # Place armies head
        # Distribution over the nodes
        self.num_place_layers = num_place_layers
        self.placeArmies_layers = torch.nn.ModuleList([ResGCN(GCNConv(hidden_conv_dim,hidden_place_dim),
                                                nn.BatchNorm1d(hidden_place_dim))]+\
                                                [ResGCN(GCNConv(hidden_place_dim,hidden_place_dim),
                                                nn.BatchNorm1d(hidden_place_dim)) for i in range(num_place_layers-2)] +\
                                               [GCNConv(hidden_place_dim, out_place_dim)]
                                               )
        # self.global_to_place = nn.Linear(hidden_global_dim, out_place_dim)
        # self.place_final_1 = nn.Linear(2*out_place_dim, out_place_dim)
        # self.place_final_2 = nn.Linear(out_place_dim, 1)

        self.place_final = torch.nn.ModuleList([nn.Linear(num_nodes, 64), nn.Linear(64, num_nodes)])
        
        # Attack head
        self.num_attack_layers = num_attack_layers
        self.hidden_attack_dim = hidden_attack_dim
        self.attack_layers = torch.nn.ModuleList([ResGCN(GCNConv(hidden_conv_dim,hidden_attack_dim),
                                                nn.BatchNorm1d(hidden_attack_dim))]+\
                                                [ResGCN(GCNConv(hidden_attack_dim,hidden_attack_dim),
                                                nn.BatchNorm1d(hidden_attack_dim)) for i in range(num_attack_layers-1)]     
                                                )
        self.attack_edge = EdgeNet(hidden_attack_dim, 28, 3, out_attack_dim)
        # self.global_to_attack = nn.Linear(hidden_global_dim, out_attack_dim)
        # self.attack_final_1 = nn.Linear(2*out_attack_dim, out_attack_dim)
        # self.attack_final_2 = nn.Linear(out_attack_dim, 1)
        self.attack_final = torch.nn.ModuleList([nn.Linear(num_edges, 64), nn.Linear(64, num_edges+1)])
        
        # Add something to make it edge-wise
        
        # Fortify head
        self.num_fortify_layers = num_fortify_layers
        self.hidden_fortify_dim = hidden_fortify_dim
        self.fortify_layers = torch.nn.ModuleList([ResGCN(GCNConv(hidden_conv_dim,hidden_fortify_dim),
                                                nn.BatchNorm1d(hidden_fortify_dim))]+\
                                                [ResGCN(GCNConv(hidden_fortify_dim,hidden_fortify_dim),
                                                nn.BatchNorm1d(hidden_fortify_dim)) for i in range(num_fortify_layers-1)]     
                                                )
        self.fortify_edge = EdgeNet(hidden_fortify_dim, 28, 3, out_fortify_dim)
        # self.global_to_fortify = nn.Linear(hidden_global_dim, out_fortify_dim)
        # self.fortify_final_1 = nn.Linear(2*out_fortify_dim, out_fortify_dim)
        # self.fortify_final_2 = nn.Linear(out_fortify_dim, 1)
        self.fortify_final = torch.nn.ModuleList([nn.Linear(num_edges, 64), nn.Linear(64, num_edges+1)])

        # Value head
        self.num_value_layers = num_value_layers
        self.value_layers = torch.nn.ModuleList([ResGCN(GCNConv(hidden_conv_dim,hidden_value_dim),
                                                nn.BatchNorm1d(hidden_value_dim))]+\
                                                [ResGCN(GCNConv(hidden_value_dim,hidden_value_dim),
                                                nn.BatchNorm1d(hidden_value_dim)) for i in range(num_value_layers-1)]
                                                )
        self.gate_nn = nn.Linear(hidden_value_dim, 1)
        self.other_nn = nn.Linear(hidden_value_dim, hidden_value_dim)
        self.global_pooling_layer = torch_geometric.nn.GlobalAttention(self.gate_nn, self.other_nn)
        self.value_fc_1 = nn.Linear(hidden_value_dim, hidden_value_dim)        
        self.value_fc_2 = nn.Linear(hidden_value_dim, 6)

        self.dropout = dropout

    def forward(self, batch, global_x):
        x, adj_t = batch.x, batch.edge_index

        
        # Global
        # for i in range(self.num_global_layers-1):
        #     global_x = self.global_fc[i](global_x)
        #     global_x = self.global_bns[i](global_x)
        #     global_x = F.dropout(F.relu(global_x), training=True, p = self.dropout) + global_x

        # global_x = self.global_fc[-1](global_x)
        

        # Initial convolution
        
        x = self.conv_init(x,adj_t)

        # Board
        for i in range(self.num_conv_layers):
            x = self.deep_convs[i](x, adj_t)  

        # pick head
        pick = self.pick_layers[0](x, adj_t)
        for i in range(1, self.num_pick_layers):
            pick = self.pick_layers[i](pick, adj_t)
        pick = pick.view(batch.num_graphs, -1)
        
        for i in range(len(self.pick_final)-1):            
            pick = F.relu(self.pick_final[i](pick))

        pick = self.pick_final[-1](pick)
        pick = F.softmax(pick, dim=1)
        
        
        # placeArmies head
        place = self.placeArmies_layers[0](x, adj_t)
        for i in range(1, self.num_place_layers):
            place = self.placeArmies_layers[i](place, adj_t)
        place = place.view(batch.num_graphs, -1)
        
        for i in range(len(self.place_final)-1):
            place = F.relu(self.place_final[i](place))

        place = self.place_final[-1](place)
        place = F.softmax(place, dim=1)
        

        # attack head
        attack = self.attack_layers[0](x, adj_t)        
        for i in range(1, self.num_attack_layers):
            attack = self.attack_layers[i](attack, adj_t)
        # Take node vectors and join them to create edge vectors

        attack = self.attack_edge.forward(attack[batch.edge_index[0]],
                                          attack[batch.edge_index[1]])
        attack = attack.view(batch.num_graphs, -1)
        for i in range(len(self.attack_final)-1):
            attack = F.relu(self.attack_final[i](attack))
        
        attack = self.attack_final[-1](attack)
        attack = F.softmax(attack, dim = 1)
        

        # fortify head
        fortify = self.fortify_layers[0](x, adj_t)        
        for i in range(1, self.num_fortify_layers):
            fortify = self.fortify_layers[i](fortify, adj_t)
        # Take node vectors and join them to create edge vectors

        fortify = self.fortify_edge.forward(fortify[batch.edge_index[0]],
                                          fortify[batch.edge_index[1]])
        fortify = fortify.view(batch.num_graphs, -1)
        for i in range(len(self.fortify_final)-1):
            fortify = F.relu(self.fortify_final[i](fortify))
            
        fortify = self.fortify_final[-1](fortify)
        fortify = F.softmax(fortify, dim = 1)

        # value head
        value = self.value_layers[0](x, adj_t)        
        for i in range(1, self.num_value_layers):
            value = self.value_layers[i](value, adj_t)
        
        # value = self.global_pooling_layer(value, batch.batch)
        # value = F.relu(self.value_fc_1(value))    

        
        value = F.relu(self.value_fc_1(value.mean(axis=0)))
        
        
        value = torch.sigmoid(self.value_fc_2(value))

        
        #########################################        
        return pick, place, attack, fortify, value
        

def TPT_Loss(output, target):    
    return -(target*torch.log(output)).sum()

def V_Loss(v, z):
    return (-z*torch.log(v) - (1-z)*torch.log(1-v)).sum()

def total_Loss(output, v, target, z):
    return TPT_Loss(output, target) + V_Loss(v,z)  
     
def train_model(model, optimizer, scheduler, criterion, device, epochs,
                train_loader, val_loader = None, eval_every = 1,
                load_path = None, save_path = None):
    ''' Train a torch_geometric model
    '''
    # Load model if load_path is given
    if not load_path is None:
        state_dict = load_dict(load_path, device, encoding = 'latin1')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        starting_epoch, best_loss = state_dict['epoch'], state_dict['best_loss']
    else:
        starting_epoch, best_loss = 0.0, 0.0

    for e in range(epochs):
        epoch = starting_epoch + e
        model.train()

        total_loss = 0
        for batch, global_batch in train_loader:   
            batch.to(device)
            global_batch.to(device) 
            optimizer.zero_grad()
            try:
                pick, place, attack, fortify, value = model.forward(batch, global_batch)
            except Exception as e:
                print("Batch: \n", batch.x)
                print(batch.x.shape)
                print("global: ", global_batch.shape)
                raise e
            # print("pick", pick)
            # print("place", place)
            # print("attack", attack)
            # print("fortify", fortify)
            # print("value", value)
            phase = batch.phase[0]
            if phase == 'initialPick':
                out = pick
            elif phase in ['initialFortify', 'startTurn']:
                out = place
            elif phase == 'attack':
                out = attack
            elif phase == 'fortify':
                out = fortify            
            
            y = batch.y.view(batch.num_graphs,-1)
            z = batch.value.view(batch.num_graphs,-1)
            # print("out\n", out)
            # print("y\n", y)
            loss = criterion(out, value, y, z)  
            # print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if not val_loader is None and (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0                
                for val_batch, val_global_batch in val_loader:
                    val_batch.to(device)
                    val_global_batch.to(device)
                    pick, place, attack, fortify, value = model.forward(val_batch, val_global_batch)
                    phase = val_batch.phase[0]
                    if phase == 'initialPick':
                        out = pick
                    elif phase in ['initialFortify', 'startTurn']:
                        out = place
                    elif phase == 'attack':
                        out = attack
                    elif phase == 'fortify':
                        out = fortify
                    y = val_batch.y.view(val_batch.num_graphs,-1)
                    loss += criterion(out, y)
                    val_loss += loss.item()
                print('Epoch: {0}, \t train_loss: {1:.3f}, \t val_loss: {2:.3f}'.format(epoch, total_loss, val_loss))
                if total_loss < best_loss:
                    save_dict(save_path, {'model':model.state_dict(),
                                          'optimizer':optimizer.state_dict(),
                                          'scheduler':scheduler.state_dict(),
                                          'epoch': epoch, 'best_loss':total_loss})
    
    # End
    save_dict(save_path, {'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'epoch': epoch, 'best_loss':total_loss})
    
    
if __name__ == "__main__":
    # Test the model loading, the forward and the loss
    print("script model.py")
    
    