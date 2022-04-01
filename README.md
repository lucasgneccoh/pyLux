# pyLux
Python implementation of Risk based on Lux Deluxe (Sillysoft software) for academic purposes

# Play with the GUI

<img src="https://github.com/lucasgneccoh/pyLux/blob/main/images/GUI_init.PNG" width="768" height="432">

## Setup

To play with the GUI, you need two things:

  - **map file**: This is a JSON file that describes the world. It contains the continents with their bonuses, names, colors and the countries they include. The countries have their names, id, associated card type, and xy coordinates to draw them in the GUI. There is also a list of links telling which countries are neighbors. See [this file](https://github.com/lucasgneccoh/pyLux/blob/main/support/maps/classic_world_map.json) for an example of the final JSON file. [This](https://github.com/lucasgneccoh/pyLux/blob/main/core/generate_classic_map.py) other script can help you create such a file.
 
  - **GUI config file**: This file is also a JSON file telling different GUI visual preferences as well as game preferences. You can change the players, the map to be loaded (related to the **map file**), the colors to use, and even screen sizes and positions of things. Be careful with those last parameters! Here is an [example](https://github.com/lucasgneccoh/pyLux/blob/main/support/GUI_config/classic_map_default_config.json) file, and again a script to [generate](https://github.com/lucasgneccoh/pyLux/blob/main/core/generate_GUI_config_file.py) it.
 
If you don't want to create or change this files yourself, the defaults will be just fine. They include the usual world map for risk. I would just recommend changing the players (default has one human against some random bots). To do so, change the *players* and *players_names* lists in the config file.

More agents will be available soon, for now you can play against another human, or play against some random, weak bots.

## Launch the GUI
Once you are ready to play, just use the following commands to go to the *core* folder and execute the GUI. (I assume you are on the root directory)

```
cd core
python GUI.py
```

You can pass some arguments to `GUI.py`. Use the command `python GUI.py --help` to get more information.

## Some instructions
Here are the commands you can use to play the game with the default configuration:

  - The main interaction happens through clicking. Click to select a country to fortify, a country to attack from and then the one to attack, etc. If you don't click ultra fast all over the screen, the game should run just fine.
  - There are buttons over the message box on the right bottom corner. This buttons allow you to change the game phase. 
    - Once you are done attacking, click on *fortify* to pass to the fortification phase.
    -  When you are ready to end your turn, click on *End turn*. 
    - The button *Cards* will show you the cards you have. They appear over the country they correspond to, and wildcards appear on the top right corner. 
    - *Cash cards* will try to cash a card set
  - If you prefer pressing keys, there are keys that do the same as the buttons (except for the card cash). You can change this keys on the preferences file.
    - Press `f` to fortify
    - Press `q` to end the turn
    - Press `c` to see the cards
  - There are some keys to make the game go faster. They are used to attack until either you conquest or get defeated, and to move troops 5 at a time, or all possible at a time.
    - Hold `Shift` while attacking to *attack till dead*
    - Hold `Ctrl` while placing armies or fortifying to place/move all available armies
    - Hold `Alt` while placing armies or fortifying to place/move 5 armies


# TODO

  - List the dependencies and make something to install everything that is needed
