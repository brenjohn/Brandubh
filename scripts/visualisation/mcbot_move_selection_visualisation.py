#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 19:56:59 2023

@author: john

This script will get a MCTSBot to build a search tree for a given board
position and generate plots showing the search statistics in the tree which are
used for deciding which branches of the tree to explore next and which move to
take.

Plots are saved to a "mc_search_stats/" directory.
"""
import os
import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from brandubh.game import GameSet, GameState
from brandubh.bots.mcbot import MCTSBot, MCTSNode


def get_stats(root, moves, player):
    win_frac = {}
    for move in moves:
        if root.has_child(move):
            win_frac[move] = root.get_child(move).winning_frac(player)
        else:
            win_frac[move] = 0 
    return win_frac


def plot_stats(win_frac, iteration, out_dir):
    plt.figure()
    
    barwidth = 0.35
    x = np.arange(len(win_frac))
    x_ticks = ["({0}, {1}, {2}, {3})".format(*k) for k in win_frac.keys()]
    
    plt.bar(x, win_frac.values(), barwidth, color='b', label='win fraction')
    plt.xticks(x, labels=x_ticks, rotation=90, fontsize=5)
    # plt.yticks([0.0, 0.5, 1.0])
    plt.xlabel('Moves')
    plt.ylabel('Win fraction')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title('Iteration {0}'.format(iteration))
    
    plt.savefig(out_dir+'{0}.png'.format(iteration), 
                dpi=300,
                bbox_inches='tight')



# %% Set up a board position to work with.
player = 1
board_pos =[[ 0, 0, 0,-1, 0, 0, 0],
            [ 0, 0, 0,-1, 0, 0, 0],
            [ 0,-1, 0, 1, 1, 0, 0],
            [ 0,-1, 2, 0,-1, 0, 0],
            [ 0, 0, 0,-1, 0,-1, 0],
            [ 0, 0, 1, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0]]
board = {}
for i in range(7):
    for j in range(7):
        board[(i, j)] = board_pos[i][j]
game_set = GameSet()
game_set.set_board(board)
game_state = GameState(game_set, player)



# %% Create a bot to work with.
bot = MCTSBot(num_rounds=14, temp=1.1, use_greedy_rand=True)


    
# %% Get the bot to build a search tree and generate plots.
moves = game_state.legal_moves()
bot.root = MCTSNode(game_state.copy())
stats = [get_stats(bot.root, moves, player)]

for i in range(200):
    bot.select_move(game_state)
    stats.append(get_stats(bot.root, moves, player))

out_dir = "mc_search_stats/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
for wf, i in zip(stats, range(len(stats))):
    plot_stats(wf, i, out_dir)