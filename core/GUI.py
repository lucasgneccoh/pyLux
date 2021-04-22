import sys
import json
import time
import copy
import pyRisk
import agent
# For the GUI
import pygame
import agent
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--preferences", help="Path to the preferences file. It is a JSON file containing all the game and visual preferences", default = "../support/GUI_config/classic_map_default_config.json")
  
  parser.add_argument("--console_debug", help="Set to true to print on the console messages to help debugging", default ="false")
  
  
  args = parser.parse_args()
  
  # Some formatting
  args.console_debug = (args.console_debug.lower()=='true')
  return args

def color_all(boxes, color=pygame.Color(211, 230, 225, a=125)):
  for i, b in boxes.items():
    b.fill(color)

def color_continents(boxes_back, board):
  for i, b in boxes_back.items():
    cont_code = board.world.countries[i].continent
    cont = board.world.continents[cont_code]
    b.fill(pygame.Color(*cont.color))

def color_countries(boxes, board, player_colors):
  for i, b in boxes.items():
    b.fill(player_colors[board.world.countries[i].owner])

def show_text(boxes, board, player_colors, W, H):
  for i, b in boxes.items():
    c = board.world.countries[i]
    msgs = []
    msgs.append(c.id)
    msgs.append(str(c.armies))    
    num_lines = len(msgs)
    for i, msg in enumerate(msgs):
      label = myfont.render('{: ^15}'.format(msg), 1, color_text, player_colors[c.owner])
      text_w, text_h = label.get_width(), label.get_height()       
      pos = (W-text_w)//2
      b.blit(label, (pos, H//2 - (num_lines//2 - i)*text_h))
        
def reset_boxes_text(boxes, player_colors, W, H):  
  for i, b in boxes.items():
    c = board.world.countries[i]
    msgs = ['','','']   
    num_lines = len(msgs)
    for i, msg in enumerate(msgs):
      label = myfont.render('{: ^15}'.format(msg), 1, color_text, player_colors[c.owner])
      text_w = label.get_width()
      
      text_h = label.get_height()
      pos = (W-text_w)//2
      b.blit(label, (pos, H//2 - (num_lines//2 - i)*text_h))
      
def show_text_cards(boxes, board, player_colors, player, wildcard_containers, color_none, color_cards, color_text, color_background, W, H):  
  color_all(wildcard_containers, color_background)
  codes = [c.code for c in player.cards]
  kind_names = {1:'Soldier',2:'Horse',3:'Cannon'}
  #print(codes)
  for i, b in boxes.items():
    c = board.world.countries[i]
    msgs = []
    if c.code in codes:
      # Show card
      #print("Show card") 
      card = player.cards[codes.index(c.code)]
      msgs.append(c.id)
      msgs.append(f'{kind_names[card.kind]}')
      color = color_cards
      msgs.append('')
    else:
      # Show nothing (empty)
      msgs = ['','','']
      color = color_none
    num_lines = len(msgs)
    for i, msg in enumerate(msgs):
      label = myfont.render('{: ^15}'.format(msg), 1, color_text, color)
      text_w, text_h = label.get_width(), label.get_height()
      pos = (W-text_w)//2
      b.blit(label, (pos, H//2 - (num_lines//2 - i)*text_h))
  
  wildcards = [card for card in player.cards if card.kind == 0]
  if len(wildcards)>0:
    # Show the wildcards in the containers
    #print("Wildcards: ", wildcards)
    for card, b in zip(wildcards, wildcard_containers.values()):
      b.fill(player_colors[player.code])
      msgs = []
      msgs.append('Wildcard')      
      num_lines = len(msgs)
      for i, msg in enumerate(msgs):
        label = myfont.render('{:<15}'.format(msg), 1, color_text, player_colors[player.code])
        text_h = label.get_height()        
        b.blit(label, (0, text_h//2))
     
def show_buttons(buttons, color_text, color_background):
  for name, b in buttons.items():    
    msgs = []    
    msgs.append(name)
    num_lines = len(msgs)
    for i, msg in enumerate(msgs):
      label = myfont.render('{:<15}'.format(msg), 1, color_text, color_background)
      text_w = label.get_width()      
      text_h = label.get_height()
      pos_x, pos_y = 0, 0
      b.blit(label, (pos_x, pos_y))
         
def show_some_message(message_box, board, color_text, color_background, msgs):  
  p = board.activePlayer  
  num_lines = len(msgs)
  for i, msg in enumerate(msgs):
    label = myfont.render('{: <80}'.format(msg), 1, color_text, color_background)    
    text_h = label.get_height()
    pos_x, pos_y = 0, (text_h)*i
    message_box.blit(label, (pos_x, pos_y))

def draw_lines(coords, board):
  ala, kam = board.getCountryById('ALA').code, board.getCountryById('KAM').code
  for e in board.world.map_graph.edges():
    # For now the Alaska-Kamchatka line is monkey patched.
    # Must fix that, insert it in the map definition
    if (e[0]==ala and e[1]==kam) or (e[1]==ala and e[0]==kam):
      # Special lines
      x,y = coords[ala]
      pygame.draw.line(screen, pygame.Color('black'), [x+SQ_RADIUS,y+SQ_RADIUS], [x-SQ_RADIUS,y+SQ_RADIUS], 6)
      x,y = coords[kam]
      pygame.draw.line(screen, pygame.Color('black'), [x+SQ_RADIUS,y+SQ_RADIUS], [x+3*SQ_RADIUS,y+SQ_RADIUS], 6)
      
    else:
      pygame.draw.line(screen, pygame.Color('black'), [x+SQ_RADIUS for x in coords[e[0]]], [x+SQ_RADIUS for x in coords[e[1]]], 6)
        
def draw_lines_attack(coords, board, country):
  ala, kam = board.getCountryById('ALA').code, board.getCountryById('KAM').code
  x, y = coords[country.code]
  for c in country.successors():
    if country.owner == c.owner:
      continue
    ex, ey = coords[c.code]
    # For now the Alaska-Kamchatka line is monkey patched.
    # Must fix that, insert it in the map definition
    if (country.code==ala and c.code==kam):
      # ALA
      pygame.draw.line(screen, pygame.Color('red'), [x+SQ_RADIUS,y+SQ_RADIUS], [x-SQ_RADIUS,y+SQ_RADIUS], 6)
      # KAM
      pygame.draw.line(screen, pygame.Color('red'), [ex+SQ_RADIUS,ey+SQ_RADIUS], [ex+3*SQ_RADIUS,ey+SQ_RADIUS], 6)
    elif (c.code==ala and country.code==kam):
      # ALA
      pygame.draw.line(screen, pygame.Color('red'), [ex+SQ_RADIUS,ey+SQ_RADIUS], [ex-SQ_RADIUS,ey+SQ_RADIUS], 6)
      # KAM
      pygame.draw.line(screen, pygame.Color('red'), [x+SQ_RADIUS,y+SQ_RADIUS], [x+3*SQ_RADIUS,y+SQ_RADIUS], 6)
    else:
      pygame.draw.line(screen, pygame.Color('red'),
      [x+SQ_RADIUS, y+SQ_RADIUS], [ex+SQ_RADIUS, ey+SQ_RADIUS], 6)
    
def draw_lines_fortify(coords, board, country):
  ala, kam = board.getCountryById('ALA').code, board.getCountryById('KAM').code
  x, y = coords[country.code]
  for c in country.successors():
    if country.owner != c.owner:
      continue
    ex, ey = coords[c.code]
    # For now the Alaska-Kamchatka line is monkey patched.
    # Must fix that, insert it in the map definition
    if (country.code==ala and c.code==kam):
      # ALA
      pygame.draw.line(screen, pygame.Color('green'), [x+SQ_RADIUS,y+SQ_RADIUS], [x-SQ_RADIUS,y+SQ_RADIUS], 6)
      # KAM
      pygame.draw.line(screen, pygame.Color('green'), [ex+SQ_RADIUS,ey+SQ_RADIUS], [ex+3*SQ_RADIUS,ey+SQ_RADIUS], 6)
    elif (c.code==ala and country.code==kam):
      # ALA
      pygame.draw.line(screen, pygame.Color('green'), [ex+SQ_RADIUS,ey+SQ_RADIUS], [ex-SQ_RADIUS,ey+SQ_RADIUS], 6)
      # KAM
      pygame.draw.line(screen, pygame.Color('green'), [x+SQ_RADIUS,y+SQ_RADIUS], [x+3*SQ_RADIUS,y+SQ_RADIUS], 6)
    else:
      pygame.draw.line(screen, pygame.Color('green'),
      [x+SQ_RADIUS, y+SQ_RADIUS], [ex+SQ_RADIUS, ey+SQ_RADIUS], 6)

def click_on_box(coords, location, SIDE):
  a, b = location    
  for i, c in coords.items():
    x, y = c[0], c[1]
    x_min, x_max = x, x+SIDE
    y_min, y_max = y, y+SIDE
    if a<x_max and a>x_min and b<y_max and b>y_min:
      return i
  return None
  
def click_on_button(coords_buttons, location, BUTTON_W, BUTTON_H):
  a, b = location    
  for i, c in coords_buttons.items():
    x, y = c[0], c[1]
    x_min, x_max = x, x+BUTTON_W
    y_min, y_max = y, y+BUTTON_H
    if a<x_max and a>x_min and b<y_max and b>y_min:
      return i
  return None
      

def print_players(board):
  print('------ PLAYERS ----------')
  armies = {i: 0 for i in board.players}
  armies[-1] = 0
  for c in board.countries:
    armies[c.owner] += c.armies
  for i, pla in board.players.items():
    print(f'Player: {pla.name_string}, inc {pla.income}, initialArmies {pla.initialArmies}, total: {armies[pla.code]}')
    
def print_countries(board):
  print('------ COUNTRIES ----------')
  for c in board.countries:
    print(f'{c.id}\t{c.armies}')
  
def shown_big_message(screen, color_text, color_text_back, WIDTH, HEIGHT):
  msgs = []
  
  msgs.append("*")
  msgs.append("       GAME OVER     ")
  msgs.append("   Close the window   ")
  msgs.append("*")
  num_lines = len(msgs)
  for i, msg in enumerate(msgs):
    label = myfont.render('{:*^50}'.format(msg), 1, color_text, color_text_back)
    text_w = label.get_width()
    
    text_h = label.get_height()
    pos = (WIDTH-text_w)//2
    screen.blit(label, (pos, HEIGHT//2 - (num_lines//2 - i)*text_h))
  pygame.display.update()

def wait_for_quit():
  for e in pygame.event.get():
    if e.type == pygame.QUIT:
      pygame.quit()
      sys.exit()


def show_active_player_box(player, player_name_containers, player_colors, color_text, color_background):  
  color_player = player_colors[player.code]
  for i, b in player_name_containers.items():
    if i==player.code:
      label = myfont.render('{:<15}'.format(player.name()), 1, color_text, color_player)
      text_h =  label.get_height() 
      b.fill(color_player)      
      b.blit(label, (0, 0))      
    else:
      b.fill(color_background)


def fill_and_label(button, color, text, color_text):
  label = myfont.render(text, 1, color_text)
  button.fill(color)
  button.blit(label, (0,0))
  

args = parseInputs()
console_debug = args.console_debug

with open(args.preferences, 'r') as f:
  prefs = json.load(f)


# Load map
mapLoader = pyRisk.MapLoader(prefs['map_path'])
mapLoader.load_from_json()
world = pyRisk.World(mapLoader)



# Set players
players = []
available_players = agent.all_agents
for p, n in zip(prefs['players'], prefs['players_names']):
  ag = available_players[p](n)
  ag.console_debug = console_debug
  players.append(ag)

# Set board
board = pyRisk.Board(world, players)
board.setPreferences(prefs)


# Draw game as a simple graph
# Screen size and country box size
WIDTH = prefs['screen_W']
HEIGHT = prefs['screen_H']
SQ_RADIUS = prefs['sq_radius']
CONTINENT_BOX_PAD = prefs['continent_box_pad']

# Position and size of info box (buttons and messages)
INFO_BOX_X = prefs['info_box_x']
INFO_BOX_Y = prefs['info_box_y']
BUTTON_W = prefs['button_w']
BUTTON_H = prefs['button_h']
BUTTON_PAD = prefs['button_pad']
MSGBOX_W = (BUTTON_W + BUTTON_PAD)*(4)- BUTTON_PAD
MSGBOX_H = BUTTON_H*3

# Wildcard containers
WILDCARD_X = prefs['wildcard_x']
WILDCARD_Y = prefs['wildcard_y']
WILDCARD_PAD = prefs['wildcard_pad']
WILDCARD_W = prefs['wildcard_w']
WILDCARD_H = prefs['wildcard_h']

FONT_SIZE = prefs['font_size']
FONT = prefs['font']

color_background = pygame.Color(*prefs['color_background'])  
color_p1 = pygame.Color(*prefs['color_p1'])  
color_p2 = pygame.Color(*prefs['color_p2'])  
color_p3 = pygame.Color(*prefs['color_p3'])  
color_p4 = pygame.Color(*prefs['color_p4'])  
color_p5 = pygame.Color(*prefs['color_p5'])  
color_p6 = pygame.Color(*prefs['color_p6'])  


color_msgbox = pygame.Color(*prefs['color_msgbox']) 
color_activated = pygame.Color(*prefs['color_activated'])
color_empty = pygame.Color(*prefs['color_empty'])  
color_text = pygame.Color(*prefs['color_text']) 

color_text_back = color_empty

player_colors = {} 
player_colors[0] = color_p1
player_colors[1] = color_p2
player_colors[2] = color_p3
player_colors[3] = color_p4
player_colors[4] = color_p5
player_colors[5] = color_p6
player_colors[-1] = color_empty

# Start pygame    
pygame.init()
pygame.display.set_caption("pyRisk")
screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
myfont = pygame.font.SysFont(FONT, FONT_SIZE)

# Create the surfaces for each country
coords = {}
max_x, max_y = 0,0
for i, c in world.countries.items():
  aux = list(map(float,c.xy.split(',')))
  if aux[0]>max_x: max_x = aux[0]
  if aux[1]>max_y: max_y = aux[1]
  coords[i] = aux

# Create the boxes representing countries to interact later

boxes_back = {}
boxes = {}
SMALL_SIDE = 2*SQ_RADIUS
BIG_SIDE = 2*(SQ_RADIUS+CONTINENT_BOX_PAD)
for i, c in coords.items():
  x, y = int(c[0]), int(c[1])
  boxes_back[i] = screen.subsurface(
      pygame.Rect(x, y, BIG_SIDE, BIG_SIDE)
      )
  boxes[i] = boxes_back[i].subsurface(
      pygame.Rect(CONTINENT_BOX_PAD, CONTINENT_BOX_PAD, SMALL_SIDE, SMALL_SIDE)
      )

# Define buttons
buttons = {}    
coords_buttons = {'Fortify':    [INFO_BOX_X + (BUTTON_W + BUTTON_PAD)*0, INFO_BOX_Y],
                  'End turn':   [INFO_BOX_X + (BUTTON_W + BUTTON_PAD)*1, INFO_BOX_Y],
                  'Cards':      [INFO_BOX_X + (BUTTON_W + BUTTON_PAD)*2, INFO_BOX_Y],
                  'Cash cards': [INFO_BOX_X + (BUTTON_W + BUTTON_PAD)*3,INFO_BOX_Y]}
for name, xy in coords_buttons.items():
  buttons[name] = screen.subsurface(
      pygame.Rect(xy[0], xy[1], BUTTON_W, BUTTON_H)
      )


# Define wildcard containers for card show phase
wildcard_containers = {}    
coords_wildcards = {1: [WILDCARD_X, WILDCARD_Y + (WILDCARD_H+WILDCARD_PAD)*0],
                    2:[WILDCARD_X,  WILDCARD_Y + (WILDCARD_H+WILDCARD_PAD)*1],
                    3:[WILDCARD_X,  WILDCARD_Y + (WILDCARD_H+WILDCARD_PAD)*2],
                    4:[WILDCARD_X,  WILDCARD_Y + (WILDCARD_H+WILDCARD_PAD)*3]}
for i, xy in coords_wildcards.items():
  wildcard_containers[i] = screen.subsurface(
      pygame.Rect(xy[0], xy[1], WILDCARD_W, WILDCARD_H)
      )


# Define message box
message_box = screen.subsurface(
      pygame.Rect(INFO_BOX_X, INFO_BOX_Y + BUTTON_PAD, MSGBOX_W , MSGBOX_H)
      )
    

# Boxes to show active player turn 
player_name_containers = {}   
PLAYER_NAME_X = 20 
PLAYER_NAME_Y = INFO_BOX_Y + BUTTON_PAD + 2*BUTTON_H 
PLAYER_NAME_W = BUTTON_W + 30
PLAYER_NAME_H = BUTTON_H
PLAYER_NAME_PAD = 10 
coords_player_name = {i: [PLAYER_NAME_X + (PLAYER_NAME_W+PLAYER_NAME_PAD)*(i), PLAYER_NAME_Y] for i in range(6)}

for i, xy in coords_player_name.items():
  player_name_containers[i] = screen.subsurface(
      pygame.Rect(xy[0], xy[1], PLAYER_NAME_W, PLAYER_NAME_H)
      )


coords_buttons_keys = {
  'till_dead': [PLAYER_NAME_X, PLAYER_NAME_Y - (BUTTON_H+10)],
  'move_5': [PLAYER_NAME_X, PLAYER_NAME_Y - (BUTTON_H+10)*2],
  'move_all': [PLAYER_NAME_X, PLAYER_NAME_Y - (BUTTON_H+10)*3],
}
buttons_keys_containers = {}
for n, xy in coords_buttons_keys.items():
  buttons_keys_containers[n] = screen.subsurface(
      pygame.Rect(xy[0], xy[1], BUTTON_W, BUTTON_H)
      )
  
    
# All start in gray
screen.fill(color_background)
draw_lines(coords, board)
color_all(boxes, color_empty)
color_all(buttons_keys_containers, color_msgbox)
for n, b in buttons_keys_containers.items():
  fill_and_label(b, color_msgbox, n, color_text)
color_all(wildcard_containers, color_background)

board.play()
color_continents(boxes_back, board)
color_countries(boxes, board, player_colors)
show_text(boxes, board, player_colors, 2*SQ_RADIUS, 2*SQ_RADIUS)
show_buttons(buttons, color_text, color_msgbox)
show_some_message(message_box, board, color_text, color_msgbox, ['',''])


pygame.display.update()

source, target, action_msg = None, None, None
running = True
showing_cards, game_over, AI_played = False, False, False 

# Define the keys to be used (internal and pygame ones)

bool_till_dead, key_till_dead = False, prefs['key_till_dead']
bool_move_5, key_move_5 = False, prefs['key_move_5']
bool_move_all, key_move_all = False, prefs['key_move_all']

key_fortify_phase = prefs['key_fortify_phase']
key_end_turn = prefs['key_end_turn']
key_cards = prefs['key_cards']

TIME_SLEEP_AI = prefs['time_sleep_ai']

last_AI_player = None
while running:
  
  msgs = []    
  
  if game_over:      
    shown_big_message(screen, color_text, color_text_back, WIDTH, HEIGHT)
    wait_for_quit()
  
  else:
    # Human player active
    if board.activePlayer.human:
    
      # More monkey patching 
      if board.gamePhase=='initialFortify' and board.activePlayer.initialArmies==0 and board.activePlayer.income==0:
        board.play()
        AI_played = True
        
      # Check if more than 5 cards at the start of the turn
      if board.gamePhase == 'startTurn' and len(board.activePlayer.cards)>=5:
        card_set = pyRisk.Deck.yieldBestCashableSet(board.activePlayer.cards, board.activePlayer.code, board.countries)
        if not card_set is None:
          armies = board.cashCards(*card_set)
          if console_debug: print(f"Force cashed {armies} armies")
          card_set = None
          action_msg = f"Forced card cash: got {armies} armies"
          c_int = None
          continue
        else:
          if console_debug: print("Cash not possible")
          action_msg = "ERROR: in forced cash human"
          running = False
      
      
      # EVENT CHECKING
      # If not, we check for the events        
      for e in pygame.event.get():
      
        if e.type == pygame.QUIT:                    
          running = False                  
          pygame.quit()
          sys.exit()
          
        elif e.type == pygame.KEYDOWN:
          if e.key == pygame.K_ESCAPE:
            #Escape key quits the game
            running = False       
            pygame.quit()
            sys.exit()
          
          elif e.key in key_till_dead:
            bool_till_dead = True            
            fill_and_label(buttons_keys_containers['till_dead'], color_activated, 'till_dead', color_text)
          elif e.key in key_move_5:
            bool_move_5 = True
            fill_and_label(buttons_keys_containers['move_5'], color_activated, 'move_5', color_text)
            continue
          elif e.key in key_move_all:
            bool_move_all = True
            fill_and_label(buttons_keys_containers['move_all'], color_activated, 'move_all', color_text)
            continue
            
          elif e.unicode == key_fortify_phase:
            # Can only do this afterplacing armies
            if board.activePlayer.income > 0:
              if console_debug: print('Can only fortify after placing armies')
              action_msg = "Place armies before fortify!"
              continue
            elif 'initial' in board.gamePhase:
              if console_debug: print('Still in initial phases')
              action_msg = "Key not available during initial phases!"
              continue
            else:
              board.gamePhase = 'fortify'
              board.updateMovable()
              source, target = None, None
              showing_cards = False
              action_msg = None
              draw_lines(coords, board)
            
              
              
          elif e.unicode == key_end_turn:
            # Can only do this afterplacing armies
            if board.activePlayer.income > 0:
              if console_debug: print('Can only end turn after placing armies')
              action_msg = "Place armies before ending turn!"
              continue
            elif 'initial' in board.gamePhase:
              if console_debug: print('Still in initial phases')
              action_msg = "Key not available during initial phases!"
              continue
            else:
              source, target = None, None
              board.gamePhase = 'end'
              showing_cards = False
              action_msg = None
              draw_lines(coords, board)
            
          elif e.unicode == key_cards:
            action_msg = None
            showing_cards = not showing_cards
            #if console_debug: print(board.activePlayer.cards)
            if showing_cards:
              show_text_cards(boxes, board, player_colors, board.activePlayer, wildcard_containers, color_empty, color_msgbox, color_text, color_background, 2*SQ_RADIUS, 2*SQ_RADIUS)          
            else:
              color_all(wildcard_containers, color_background)
          else:
            # Reset the message if one is being shown
            action_msg = None
          
        elif e.type == pygame.KEYUP:          
            
          if e.key in key_till_dead:
            bool_till_dead = False            
            fill_and_label(buttons_keys_containers['till_dead'], color_msgbox, 'till_dead', color_text)
          elif e.key in key_move_5:
            bool_move_5 = False              
            fill_and_label(buttons_keys_containers['move_5'], color_msgbox, 'move_5', color_text)
          elif e.key in key_move_all:
            bool_move_all = False               
            fill_and_label(buttons_keys_containers['move_all'], color_msgbox, 'move_all', color_text)
          
          
        elif e.type == pygame.MOUSEBUTTONDOWN:
          # Determine what user clicked
          location = e.pos
          c_name = click_on_button(coords_buttons, location, BUTTON_W, BUTTON_H)
          
          ##### Actions when clicking on button
          if not c_name is None:
            #if console_debug: print(f"Clicked on {c_name}")
            if c_name == 'Fortify':
              # Can only do this afterplacing armies
              if board.activePlayer.income > 0:
                if console_debug: print('Can only fortify after placing armies')
                action_msg = "Place armies before fortify!"
                continue
              elif 'initial' in board.gamePhase:
                if console_debug: print('Still in initial phases')
                action_msg = "Key not available during initial phases!"
                continue
              else:
                board.gamePhase = 'fortify'
                board.updateMovable()
                source, target = None, None
                showing_cards = False
                action_msg = None
                draw_lines(coords, board)
                
            
            elif c_name == 'End turn':
              # Can only do this afterplacing armies
              if board.activePlayer.income > 0:
                if console_debug: print('Can only end turn after placing armies')
                action_msg = "Place armies before ending turn!"
                continue
              elif 'initial' in board.gamePhase:
                if console_debug: print('Still in initial phases')
                action_msg = "Key not available during initial phases!"
                continue
              else:
                source, target = None, None
                board.gamePhase = 'end'
                showing_cards = False
                action_msg = None
                draw_lines(coords, board)
                
            elif c_name == 'Cards':
              action_msg = None
              showing_cards = not showing_cards
              #if console_debug: print(board.activePlayer.cards)
              if showing_cards:
                show_text_cards(boxes, board, player_colors, board.activePlayer, wildcard_containers, color_empty, color_msgbox, color_text, color_background, 2*SQ_RADIUS, 2*SQ_RADIUS)
              else:
                color_all(wildcard_containers, color_background)
                
            elif c_name == 'Cash cards':
              if board.gamePhase != "startTurn":
                if console_debug: print("Cash not possible: Only starting the turn")
                action_msg = 'No cash after turn start phase!'
                continue
              showing_cards = False
              card_set = pyRisk.Deck.yieldBestCashableSet(board.activePlayer.cards, board.activePlayer.code, board.countries)
              if not card_set is None:
                armies = board.cashCards(*card_set)
                if console_debug: print(f"Cashed {armies} armies")
                board.gamePhase = 'startTurn'
                card_set = None
                action_msg = f'Cashed {armies} armies'
              else:
                action_msg = 'Cash not possible'
                if console_debug: print("Cash not possible")
              

          if showing_cards: continue
          
          
          c_int = click_on_box(coords, location, 2*SQ_RADIUS)
          
          if c_int is None: continue
          c = board.world.countries[c_int]
          
          if console_debug: print(f"Clicked on {c.id}")
          
          ##### Actions when clicking on countries
          
          if board.gamePhase == 'initialPick':
            if c.owner == -1:
              if console_debug: print(f'Human picked: {c.id}')
              board.initialPickOneHuman(c)
              action_msg = f'Picked {c.name}'
            else:
              action_msg = 'Territory already has an owner'
              if console_debug: print("Territory already has an owner")
          
          # TODO: Why are the keys not working now?? only on second touch?? Maybe add the continue again? or a break?
          # Maybe there is a more clever way to handle events than the for all events??
          if board.gamePhase == 'initialFortify':
            if c.owner == board.activePlayer.code:
              if console_debug: print(f'Human initial fortifiy: {c.id}')
              numberOfArmies = 1
              if bool_move_all: numberOfArmies = 0
              if bool_move_5: numberOfArmies = 5
              board.initialFortifyHuman(c, numberOfArmies)
              action_msg = f'Fortified {c.name}'
            else:
              if console_debug: print("Territory belongs to enemy player")
              action_msg = "Territory belongs to enemy player"
          
          if board.gamePhase == 'startTurn':                              
            # For now, put them all on the chosen country
            if c.owner == board.activePlayer.code:
              if console_debug: print(f'Human added armies: {c.id}') 
              numberOfArmies = 1
              if bool_move_all: numberOfArmies = 0
              if bool_move_5: numberOfArmies = 5
              board.startTurnPlaceArmiesHuman(c, numberOfArmies)
            else:
              if console_debug: print("Territory belongs to enemy player")
              action_msg = "Territory belongs to enemy player"
            
          if board.gamePhase == 'attack':            
            if c.owner == board.activePlayer.code:            
              source = c
              target = None
              draw_lines(coords, board)
              draw_lines_attack(coords, board, source)
              action_msg = None
              
            elif c.owner != board.activePlayer.code:
              if not source is None and c.code in source.getHostileAdjoiningCodeList():
                # Perform attack
                target = c
                
                if console_debug: print(f'Human attacked {source.id} -> {target.id}')
                
                val = board.attack(source.code, target.code, bool_till_dead)
                battle_results = {7: 'Conquest!', 13:'Defeat...', 0:'Not much', -1:'ERROR'}                  
                res = battle_results[val]                  
                action_msg = f'Attack: {source.id} -> {target.id}. Result: {res}'
                # Reset
                
                if val == 7:
                  #print(source.id)
                  source = copy.deepcopy(target)
                  #print("Changed?",source.id)
                target = None
                draw_lines(coords, board)
                draw_lines_attack(coords, board, source)
                  
          if board.gamePhase == 'fortify':
            
            if c.owner == board.activePlayer.code and source is None:            
              source = c                                
              action_msg = None
              draw_lines(coords, board)
              draw_lines_fortify(coords, board, source)
            elif c.owner == board.activePlayer.code:
              if not source is None:
                if c == source:
                  source = None
                  draw_lines(coords, board)
                  action_msg = None
                elif c in source.successors():
                  # Fortify
                  target = c
                  if console_debug: print(f'Human fortified {source.id} -> {target.id}')
                  
                  numberOfArmies = 1
                  if bool_move_all: numberOfArmies = source.movable_armies
                  if bool_move_5: numberOfArmies = min(5, source.movable_armies)
                                      
                  board.fortifyArmies(numberOfArmies, source.code, target.code)
                  # Reset                    
                  action_msg = f'Fortified: {source.id}->{target.id}'
                  target = None
                  draw_lines(coords, board)
                  draw_lines_fortify(coords, board, source)
                  
          
          
            
        else:
          # Nothing happened: Wait
          pass
             
    # AI player      
    else:
      # This function contains all the different stages
      if console_debug: 
        print(f'Player {board.activePlayer.code}, {board.activePlayer.name()}, board phase: {board.gamePhase}')

      action_msg = f'Player {board.activePlayer.code}, {board.activePlayer.name()}'
      last_AI_player = board.activePlayer
      board.play() 
      AI_played = True
      
      

    # -----------------------------------------------
    # Update the screen
    
    if not showing_cards:
      color_all(wildcard_containers, color_background)
      color_continents(boxes_back, board)
      color_countries(boxes, board, player_colors)
      show_text(boxes, board, player_colors, 2*SQ_RADIUS, 2*SQ_RADIUS)
    
    # Need to erase messages showed to human and end turn using board.play()
    if board.activePlayer.human and board.gamePhase == 'end':        
      source, target = None, None
      draw_lines(coords, board)
      board.play()          
    
    # Messages to show next      
    msgs = []
    
    if not AI_played:
      msgs.append(f'Player {board.activePlayer.name()}: {board.gamePhase}')
      # This is in the middle of the human turn
      if board.gamePhase == 'initialPick':
        msgs.append('Pick an empty country')
      elif board.gamePhase == 'initialFortify':                  
        msgs.append(f'Armies: {board.activePlayer.income} ({board.activePlayer.initialArmies} more to use)')
      elif showing_cards:
        msgs.append(f'Next cash is worth {board.nextCashArmies}')
      elif board.gamePhase == 'startTurn':
        msgs.append(f'Armies: {board.activePlayer.income}')
      elif board.gamePhase == 'attack':
        if not source is None:
          msgs.append(f'Source: {source.id}')
      elif board.gamePhase == 'fortify':
        if not source is None:
          msgs.append(f'Source: {source.id}')
      
      
    elif AI_played:
      msgs.append('AI is playing')
      # This is just after the AI played, so player is human, but we want to show something
      # about the AI actions or status
      msgs.append('Enemy turn')
      # The name comes from before the board.play()
      
    
    
    # Action message goes last
    msgs.append('' if action_msg is None else action_msg)
    
    
    # To reset the message box
    show_some_message(message_box, board, color_text, color_msgbox, ['','',''])    
    show_some_message(message_box, board, color_text, color_msgbox, msgs)
    
    player_to_show = last_AI_player if AI_played else board.activePlayer
    show_active_player_box(player_to_show, player_name_containers, player_colors, color_text, color_background)
    
    if board.getNumberOfPlayersLeft() == 1:
      game_over = True
    
    pygame.display.update()
    
    # TODO: Red lines not dissapearing. Trouble when going to fortify phase
      # Message box needs two clicks
    
    if AI_played:
      # To see what is happening, end of the AI turn        
      time.sleep(TIME_SLEEP_AI)
      action_msg = None
      AI_played = False
  
  
  
  
  
  
  ### Starting phase: Pick countries
  
    
  
