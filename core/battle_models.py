#%% TESTING
if __name__ == '__main__':

  console_debug = True
  
  # Load map
  path = '../support/maps/classic_world_map.json'
  path = '../support/maps/test_map.json'
    
  world = World(path)
  

  # Set players
  aggressiveness = 0.7
  # TODO: Set the MCTS-neural net players
  pH, pR1, pR2 = agent.Human('PocketNavy'), agent.RandomAgent('Red',aggressiveness), agent.RandomAgent('Green',aggressiveness)
  pMC = agent.FlatMC('MC', agent.RandomAgent, budget=300)
  pMC2 = agent.FlatMC('MC', agent.RandomAgent, budget=300)
  pP = agent.PeacefulAgent()
  pA = agent.RandomAggressiveAgent('RandomAgressive')
 
  players = [pR1, pA]
  # Set board
  prefs = {'initialPhase': False, 'useCards':True,
           'transferCards':True, 'immediateCash': True,
           'continentIncrease': 0.05, 'pickInitialCountries':True,
           'armiesPerTurnInitial':4,'console_debug':False}  
  board_orig = Board(world, players)
  board_orig.setPreferences(prefs)
  board_orig.showPlayers()
  
  # Battle here. Create agent first, then set number of matches and play the games
  
  
    

