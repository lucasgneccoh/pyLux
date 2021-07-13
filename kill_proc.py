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
    
def get_col_index(columns, s):    
    for i, c in enumerate(columns):
        if s in c: return i
    return None
  
args = parseInputs()

# *** Read the file
table = pd.read_fwf(args.proc, infer_nrows=9999999)

print(table)

cont = input("Is this table correct? (y/n) Check for columns. There must be at least one column named PID, other columns are optional\nYour answer: ")

if cont.lower() != "y":
    sys.exit(0)
    
col, filt = "***", "***"
cols, filts = [], []
while col and filt:
    print("Enter information for filtering. Leave empty to quit")
    col = input("Enter the column name (or part of it) to apply a filter")
    filt = input(f"Enter the value to be looked for in the column '{col}'")
    if col and filt:
        cols.append(col)
        filts.append(filt)


# Filter the table
for col, filt in zip(cols, filts)
    # Get column index
    ind = get_col_index(table.columns, col)
    if ind is None: 
        print(f"WARNING: No filter possible for {col}")
        continue
    col_name = table.columns[ind]        
    table = table.loc[table[col_name].str.contains(filt),:]


print("\nThis is the resulting table")
print()
print(table)

cont = input("Do you wish to kill all these processes? (y/n)")

if cont.lower() != "y":
    sys.exit(0)

# Kill them
print("Kill processes here")


