import sys
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parseInputs():
  parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  parser.add_argument("--proc", help="File name with a table containing at least the PID of the processes. It may contain other also the user and the command", required=True)
  parser.add_argument("--user", help="User or list of users to filter", nargs = "*")
  parser.add_argument("--cmd", help="cmd to consider. The word will be searched in the available column CMD", nargs = "*")
  args = parser.parse_args()
  return args 
  
  
args = parseInputs()

# *** Read the file

# First line just to detect the widths
with open(args.proc, "r") as f:
    header = f.readline()
    f.close()

widths = []
curr = 0
running = False
for i, c in enumerate(header):
    if c!=" ":
        if not running:            
            running = True        
    else:
        if running:
            # Arrived to the end
            widths.append(curr)
            running = False 
            curr = 0
    curr += 1

widths.append(curr)
print(widths)
    

# Read the table
table = pd.read_fwf(args.proc, widths=widths)

print(table)

cont = input("Is this table correct? (y/n) Check for columns. There must be at least one column named PID, other columns are optional\nYour answer: ")

if cont.lower() == "y":
    print("Continue")
else:
    print("Stop")
