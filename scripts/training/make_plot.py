#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 23:41:45 2022

@author: john
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
def read_results(filename):
    file = open(filename, 'r')
    
    scores = []
    epochs = []
    # whtwin = []
    # blkwin = []
    # gsplyd = []
    
    line = file.readline()
    while line:
        result = line.split()
        scores.append(float(result[0]))
        epochs.append(int(result[4]))
        line = file.readline()
        
    file.close()
    return scores, epochs


#%%

filenames = ["rand_no_lookahead", 
             "rand_with_lookahead",
             "grnd_no_lookahead",
             "grnd_with_lookahead",
             # "mcts_with_lookahead_350",
             # "mcts_with_lookahead_700",
             # "mcts_with_lookahead_1050"
             ]

xs, ys = [], [] 
for filename in filenames:
    y, x = read_results(filename + ".txt")
    xs.append(x)
    ys.append(y)


plt.figure(dpi=300)
plt.plot()
plt.ylim([-1, 1])
plt.grid()
window = 8
for x, y, name in zip(xs, ys, filenames):
    cs = np.cumsum(y)
    avg_y = (cs[window:] - cs[:-window]) / float(window)
    plt.plot(x, y, label=name)
    plt.plot(x[window//2:-window//2], avg_y, color="black")
    
# plt.plot(xs[-1], ys[-1], "-o", label=filenames[-1], markersize=5)

plt.xlabel("Epochs")
plt.ylabel("Avrg. Score")
plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.savefig("random_bots_eval.png", bbox_inches="tight")