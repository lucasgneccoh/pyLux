# Starting (14-04-2021)
I have been looking at the structure of Lux by Sillysoft. It is fairly complex because they deal with online games. I want a simpler implementation, starting without any graphic interface, just to be able to play with different algorithms and train deep models (Expert iteration for example).

First things to do:

  - Define the general structure: 
    - *Lux*: Higher class. Maybe not needed here
    - *Board*: Very important. Represents the game state. In the real Lux, it contains a *World* that has the actual game state information. Maybe I could follow that idea.
    - *Agent*: Try to implement this in a general way and replicating the *LuxAgent*, so that hopefully I can use my python players in the real Lux
    - *Map*: Maps should be hackable. Better to create a file to load/create them. Lux uses an xml file to represent a map. To make it simple, I can avoid all the GUI related things and just represent a map as a network. Maybe even use `networkx` for that
    - *Project tree*: I think I will also mimic the Lux structure somehow. Folder *Support* contains maps and agents, folder *pyLux* will contain the game engine.
   

