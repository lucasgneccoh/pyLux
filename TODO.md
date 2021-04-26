# Starting (14-04-2021)
I have been looking at the structure of Lux by Sillysoft. It is fairly complex because they deal with online games. I want a simpler implementation, starting without any graphic interface, just to be able to play with different algorithms and train deep models (Expert iteration for example).

First things to do:

  - Define the general structure: 
    - *Lux*: Higher class. Maybe not needed here
    - *Board*: Very important. Represents the game state. In the real Lux, it contains a *World* that has the actual game state information. Maybe I could follow that idea.
    - *Agent*: Try to implement this in a general way and replicating the *LuxAgent*, so that hopefully I can use my python players in the real Lux
    - *Map*: Maps should be hackable. Better to create a file to load/create them. Lux uses an xml file to represent a map. To make it simple, I can avoid all the GUI related things and just represent a map as a network. Maybe even use `networkx` for that
    - *Project tree*: I think I will also mimic the Lux structure somehow. Folder *Support* contains maps and agents, folder *pyLux* will contain the game engine.
    
  - How to use Java from Python and viceversa?
    - Jpype (From python)
    - Py4J (The two must be open and running)
    - User Java Process. See C:\Users\lucas\OneDrive - Université Paris-Dauphine\S4\Stage M2\SillysoftSDK\src\com\sillysoft\lux\Gemisys.java, they call a perl script. This could be the easiest way in terms of not depending on external stuff
    - Then the output could be written somewhere and used as training data in Python.
   

# Created map (16-04-2021)
I finished the first map. To model it I used networkx, so the world is made of three basic things: a nx.DiGraph for the world, a dictionary of countries and a dictionary of continents. Classes Country and Continent extend dict, they use both the integer codes as keys. This integer country code is used to acces the world graph.
All the attributes of countries and continents are stored in the dictionaries, graph is only to get edges.
Next step:

  - Finish the Country and Continent classes.
  - Implement the Board class.
  - Tests the game
  - Implement cards

# GUI ready (22-04-2021)
GUI is ready and most of the game logic is done. Now I have to finish the GUI (add a parser for arguments, etc) and then start coding the fast mode to leave algorithms play. I also have to generalize a bit the different phases and the concept of "move" to first test the search methods. The Board class in Lux has a lot of methods, but they are very game dependent. I need some more general ones.
Almost finished. Now everything needs to be documented. Agents is already done, now pyRisk is missing.
Next step is to generalize and try first search algorithms.


# First MC method (26-04-2021)
Yesterday I restructured a bit the class organization for the project to make board copying and simulations simpler. The flat MC agent is working, but it is not quite good. At least he is now winning against the peaceful agent. It is very sensitive to some things
  - The scoring function for the resulting boards after simulation: What has worked the best are "continuous" scores. I tried simple 1/0 scores, but in the end it was acting somehow random and dumb
  - The order in which the moves are visited
  
# Java and Python (26-04-2021)
Now that the first MC was more tested, I will focus on using the Java AIs on Python or viceversa. The main idea I have right now is to comunicate the board in some way.

In Python, my board has a World with the map, countries, continents, etc.
In Java, a World object is also behind, but I have not access to it. I can not see the source code

Using Python inside Java
In python I can use everything as long as I am able to build the whole board from a board in Java.
I could run the game in Java, send an initial board to build the world and set all the players to random or dummy, and update the board everytime the player has to play just to get an answer to every call