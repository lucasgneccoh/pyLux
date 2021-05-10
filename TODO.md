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

# Making the connection (28-04-2021)
I am using conda as my Python distribution, so to be able to run python from cmd I need
  - `conda init -all` to be able to use all the shells properly
  - `conda activate [env]` before using the shell to tun python. If not, importing numpy will not work for example
To do this from Java using ProcessBuilder, the best way is to create a bat file. I guess for Linux it would be the same thing with a bash file, but for linux I guess there will be no need for conda
The idea is that the responsability for the execution of python code is tranfered to the user, who has to adapt the bat/bash file. This file will then run the python script.

There are some thing to keep in mind. If the communication is made using files, then each turn will take a lot more time. This is not so bad if we just want to test the PythonPlayer once or twice, but training a model is not a good idea. To do so, I would have to look at other options that make data sharing much faster.

I will now define the file format to transfer information. This file format will have to be common on both Python and Java side.
The idea is that the python player will just be an intermediate step, getting information from the board, passing it to Python, then receiving the action and doing it on the Java board side. It will take time, but it should work

# Making the connection II (29-04-2021)
It works. I am able to play Lux using a Java player that gets all his moves from running a python script an getting back the output.
For now, I create a file to communicate the state of the game. I plan to test using the args as a command line instruction to see if it is faster.

Next step is to formalize both classes (Java and Python) and then modify my PyRisk to be the most compatible possible.

# Making the connection III (04-05-2021)
I have formalized the connection. There is a "general" Jave class and Python counterpart to use.

Now the biggest design work to do is to make PyLux more compatible with the representation of the board used in the commuunication.
As I can not pass the java Board itself, I pass to python the information about the board in some encoding I defined. I pass the continents, tue countries, the players, etc. I would like to make PyLux compatible with this representation of the GameState so that the Python players in Java are just normal PyLux players wraped with the communication class.



