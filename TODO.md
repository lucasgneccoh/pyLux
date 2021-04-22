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