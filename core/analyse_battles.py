# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:17:15 2021

@author: lucas
"""

import pandas as pd

def print_results(path):
    
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



#%%  MAIN 
paths = ["../support/battles/random_vs_net_strong.csv",
         "../support/battles/net_strong_vs_random.csv",
         "../support/battles/net_vs_net_strong_first.csv",
         "../support/battles/net_vs_net_weak_first.csv"]

for path in paths:
    print_results(path)