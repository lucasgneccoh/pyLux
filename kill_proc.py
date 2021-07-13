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

# Read the file
table = pd.read_fwf(args.proc)

print(table)
