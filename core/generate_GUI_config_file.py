#%% Imports
import json
import os
import pygame

# Define the preferences here
preferences = {
  # Game
  'map_path'  : '../support/maps/classic_world_map.json',
  'players'   : ['human', 'random'],
  'players_names'   : ['PocketNavy', 'P2'],
  'pickInitialCountries': False,
  'initialPhase': False,
  'useCards':True,
  'transferCards':True,
  'immediateCash':False,
  'continentIncrease':0.05,
  'armiesPerTurnInitial':4,
  'key_till_dead': [pygame.K_LSHIFT, pygame.K_RSHIFT],
  'key_move_5': [pygame.K_LALT, pygame.K_RALT],
  'key_move_all': [pygame.K_LCTRL, pygame.K_RCTRL],
  'key_fortify_phase': 'f',
  'key_end_turn': 'q',
  'key_cards': 'c',
  'time_sleep_ai':1,
  # Visual: Be very carefull, everything must fit well in the screen
  'screen_W':1280,
  'screen_H':720,
  'sq_radius':23,
  'continent_box_pad':10,
  'info_box_x': 930,
  'info_box_y': 640,
  'button_w': 70,
  'button_h': 14,
  'button_pad': 20,
  'wildcard_x':1200,
  'wildcard_y':20,
  'wildcard_w':70,
  'wildcard_h':20,
  'wildcard_pad':10,
  'color_background': [255, 250, 235],
  'color_p1': [91, 86, 252],
  'color_p2': [255, 105, 92],
  'color_p3': [39, 194, 37],
  'color_p4': [255, 238, 51],
  'color_p5': [250, 249, 242],
  'color_p6': [153, 0, 140],
  'color_msgbox': [255, 195, 117],
  'color_activated': [255, 28, 55],
  'color_empty': [168, 168, 145],
  'color_text': [0,0,0],
  'font': 'consolas',
  'font_size': 12,
  
}

#%% Create & save
path = '../support/GUI_config/classic_map_default_config.json'

# Create the classic world map from Risk
if os.path.exists(path):
  os.remove(path)
  
# Save the file in the json format
with open(path,'w') as f:
  json.dump(preferences, f)