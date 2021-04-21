import sys
import time
import copy
import pyRisk
import agent
# For the GUI
import pygame



def color_all(boxes, color=pygame.Color(211, 230, 225, a=125)):
  for i, b in boxes.items():
    b.fill(color)

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

#%% TESTING
if __name__ == '__main__':

  console_debug = False
  
  # Load map
  mapLoader = pyRisk.MapLoader('../support/maps/classic_world_map.json')
  mapLoader.load_from_json()
  world = pyRisk.World(mapLoader)
  
  
  
  # Set players
  
  p1, p2, p3, p4 = agent.Human('PocketNavy'), agent.RandomAgent('Blue'), agent.RandomAgent('Green'), agent.RandomAgent('Yellow')
  #p1.console_debug = console_debug  
  players = [p1, p2, p3, p4]
  # Set board
  board = pyRisk.Board(world, players)
  board.pickInitialCountries = False
  
  
  
  # Draw game as a simple graph
  # Screen size and country box size
  HEIGHT = 720
  WIDTH = 1280
  SQ_RADIUS = 25
  
  # Position and size of info box (buttons and messages)
  INFO_BOX_X = 850+80
  INFO_BOX_Y = 640
  BUTTON_W = 70
  BUTTON_H = 12
  BUTTON_PAD = 20
  MSGBOX_W = (BUTTON_W + BUTTON_PAD)*(4)- BUTTON_PAD
  MSGBOX_H = BUTTON_H*3
  
  # Wildcard containers
  WILDCARD_X = 1120+80
  WILDCARD_Y = 20
  WILDCARD_PAD = 10
  WILDCARD_W = 70
  WILDCARD_H = 20
  
  FONT_SIZE = 12
  color_background = pygame.Color(255, 250, 235)  
  color_blue = pygame.Color(127, 122, 255)
  color_red = pygame.Color(255, 100, 97)
  color_green = pygame.Color(112, 255, 136)
  color_yellow = pygame.Color(255, 250, 92)
  color_cyan = pygame.Color(112, 253, 255)
  color_pink = pygame.Color(255, 128, 253)
  
  
  color_orange = pygame.Color(255, 195, 117, a=125)
  color_gray = pygame.Color(168, 168, 145, a=125)
  color_text = pygame.Color(0,0,0, a=125)
  color_text_back = color_gray
 
  player_colors = {} 
  player_colors[0] = color_red
  player_colors[1] = color_blue
  player_colors[2] = color_green
  player_colors[3] = color_yellow
  player_colors[4] = color_cyan
  player_colors[5] = color_pink
  player_colors[-1] = color_gray
  
  # Start pygame    
  pygame.init()
  pygame.display.set_caption("pyRisk")
  screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
  myfont = pygame.font.SysFont('consolas', FONT_SIZE)
  
  # Create the surfaces for each country
  coords = {}
  max_x, max_y = 0,0
  for i, c in world.countries.items():
    aux = list(map(float,c.xy.split(',')))
    if aux[0]>max_x: max_x = aux[0]
    if aux[1]>max_y: max_y = aux[1]
    coords[i] = aux
  
  # Create the boxes representing countries to interact later
  boxes = {}
  for i, c in coords.items():
    x, y = int(c[0]), int(c[1])
    boxes[i] = screen.subsurface(
        pygame.Rect(x, y, 2*SQ_RADIUS, 2*SQ_RADIUS)
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
        

  
  # All start in gray
  screen.fill(color_background)
  draw_lines(coords, board)
  color_all(boxes)
  color_all(wildcard_containers, color_background)
  
  board.play()
  
  color_countries(boxes, board, player_colors)
  show_text(boxes, board, player_colors, 2*SQ_RADIUS, 2*SQ_RADIUS)
  show_buttons(buttons, color_text, color_orange)
  show_some_message(message_box, board, color_text, color_orange, ['',''])
  
  
  pygame.display.update()
  
  source, target, action_msg = None, None, None
  running = True
  showing_cards, game_over, AI_played = False, False, False 
  
  # Define the keys to be used (internal and pygame ones)
  
  bool_till_dead, key_till_dead = False, 'a'
  bool_move_5, key_move_5 = False, '5'
  bool_move_all, key_move_all = False, '0'
  
  key_fortify_phase = 'f'
  key_end_turn = 'q'
  key_cards = 'c'
  
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
          card_set = Deck.yieldBestCashableSet(board.activePlayer.cards, board.activePlayer.code, board.countries)
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
        # Now only working with clicks
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
            
            elif e.unicode == key_till_dead:
              bool_till_dead = True 
              print('bool_till_dead', bool_till_dead)
            elif e.unicode == key_move_5:
              bool_move_5 = True
              print('bool_move_5', bool_move_5)
            elif e.unicode == key_move_all:
              bool_move_all = True
              print('bool_move_all', bool_move_all)
              
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
              
            elif e.unicode == key_cards:
              action_msg = None
              showing_cards = not showing_cards
              #if console_debug: print(board.activePlayer.cards)
              if showing_cards:
                show_text_cards(boxes, board, player_colors, board.activePlayer, wildcard_containers, color_gray, color_text, color_orange, 2*SQ_RADIUS, 2*SQ_RADIUS)          
              else:
                color_all(wildcard_containers, color_background)
            else:
              # Reset the message if one is being shown
              action_msg = None
            
          elif e.type == pygame.KEYUP:          
              
            if e.unicode == key_till_dead:
              bool_till_dead = False
              print('bool_till_dead', bool_till_dead)
            elif e.unicode == key_move_5:
              bool_move_5 = False
              print('bool_move_5', bool_move_5)
            elif e.unicode == key_move_all:
              bool_move_all = False
              print('bool_move_all', bool_move_all)
            
            
          elif e.type == pygame.MOUSEBUTTONDOWN:
            # Determine what user clicked
            location = pygame.mouse.get_pos()
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
                  
              elif c_name == 'Cards':
                action_msg = None
                showing_cards = not showing_cards
                #if console_debug: print(board.activePlayer.cards)
                if showing_cards:
                  show_text_cards(boxes, board, player_colors, board.activePlayer, wildcard_containers, color_gray, color_text, color_orange, 2*SQ_RADIUS, 2*SQ_RADIUS)          
                else:
                  color_all(wildcard_containers, color_background)
                  
              elif c_name == 'Cash cards':
                if board.gamePhase != "startTurn":
                  if console_debug: print("Cash not possible: Only starting the turn")
                  action_msg = 'No cash after turn start phase!'
                  continue
                showing_cards = False
                card_set = Deck.yieldBestCashableSet(board.activePlayer.cards, board.activePlayer.code, board.countries)
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
              #print("ATTACK: c:{} source:{} target:{}".format(c.id, source, target))
              if c.owner == board.activePlayer.code:            
                source = c                
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
                  draw_lines_attack(coords, board, source)
                    
            if board.gamePhase == 'fortify':
              
              if c.owner == board.activePlayer.code and source is None:            
                source = c                                
                action_msg = None
                
              elif c.owner == board.activePlayer.code:
                if not source is None:
                  if c == source:
                    source = None
                    draw_lines(coords, board)
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
        board.play() 
        AI_played = True
        
        

      # -----------------------------------------------
      # Update the screen
      
      if not showing_cards:
        color_all(wildcard_containers, color_background)
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
      show_some_message(message_box, board, color_text, color_orange, ['',''])    
      show_some_message(message_box, board, color_text, color_orange, msgs)
      
      if board.getNumberOfPlayersLeft() == 1:
        game_over = True
      
      pygame.display.flip()
      
      
      
      
      if AI_played:
        # To see what is happening, end of the AI turn        
        time.sleep(0.5)
        action_msg = None
        AI_played = False
    
  
  
  
  
  
  ### Starting phase: Pick countries
  
    
  
