# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:17:15 2021

@author: lucas
"""

import pandas as pd
import os

def print_results(path, player):
    
    name = path.split("/")[-1].split(".")[0]
    print("**** Results for ", name)
    
    table = pd.read_csv(path)
    table.drop(columns = table.columns[0], inplace=True)
    
    # Detect players
    p1 = '_'.join(table.columns[1].split("_")[0:-1])
    p2 = '_'.join(table.columns[-1].split("_")[0:-1])
    
    print(f"Match:  {p1} vs {p2}")
    
    # Wins
    c1, c2 = table.loc[:, p1+"_countries"], table.loc[:, p2+"_countries"]
    
    draws = ((c1 * c2)>0).sum()
    
    w1, w2 = (c1>0).sum() - draws, (c2>0).sum() - draws
    
    total = w1+w2+draws
    
    print(f"Player 1 ({p1}) won {w1}/{total}, win rate {round(w1/total, 2)}")
    print(f"Player 2 ({p2}) won {w2}/{total}, win rate {round(w2/total, 2)}")
    
    
    # Game length
    # Not able yet
    print(" ----------------------- \n\n")
    
    winner = (c1)
    
    
    
    return table



#%%  MAIN 


root = "C:/Users/lucas/OneDrive/Documentos/stage_risk/battles"
root = "C:/Users/lucas/OneDrive/Documentos/GitHub/pyLux/support/battles"

paths = ["final_net_vs_random_first_5.csv",
         "final_net_vs_random_second_5.csv"]

for path in paths:
    table = print_results(os.path.join(root,path), "net")
    # pass
    

# path = "C:/Users/lucas/OneDrive/Documentos/stage_risk/battles/final_net_vs_random_first.csv"

# table = pd.read_csv(path)
# table.drop(columns = table.columns[0], inplace=True)
# table.drop(columns = [c for c in table.columns if "income" in c], inplace=True)

# p1 = '_'.join(table.columns[1].split("_")[0:-1])
# p2 = '_'.join(table.columns[-1].split("_")[0:-1])
# if "net" in p2:
#   aux = p2
#   p2 = p1
#   p1 = p2


# c1, c2 = table.loc[:, p1+"_countries"], table.loc[:, p2+"_countries"]
# win = (c1>0) & (c2==0)
# draw = (c1>0) & (c2>0)

# table['net_win'] = win
# table['draw'] = draw

# table

# gb = table.groupby(by=["net_win"]).mean()
# gb


