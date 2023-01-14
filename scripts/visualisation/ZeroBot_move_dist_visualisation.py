#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 21:21:59 2022

@author: john

This script will get a ZeroBot to build a search tree for a given board
position and generate plots showing the search statistics in the tree which are
used for deciding which branches of the tree to explore next and which move to
take.

Plots are saved to a "zero_search_stats/" directory.
"""
import os
import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import matplotlib.pyplot as plt

from brandubh.game import GameSet, GameState
from brandubh.bots.zero_bot.brandubh_zero import ZeroBot
from brandubh.bots.zero_bot.networks.zero_network import ZeroNet


def get_states(root, moves):
    """
    This function pulls the search stats out of the given search tree for the
    given moves.
    """
    prior_dist = {}
    visit_dist = {}
    avrg_value = {}
    puct_score = {}
    
    total_visit_count = root.total_visit_count
    
    c_sqrt_total_n = np.sqrt(total_visit_count) * 2
    
    def branch_score(move):
        q = root.expected_value(move)
        p = root.prior(move)
        n = root.visit_count(move)
        return q + p * c_sqrt_total_n/(1+n)
    
    
    for move in moves:
        if move in root.branches:
            branch = root.branches[move]
            prior_dist[move] = branch.prior
            visit_dist[move] = branch.visit_count/total_visit_count
            avrg_value[move] = root.expected_value(move)
        else:
            prior_dist[move] = 0
            visit_dist[move] = 0
            avrg_value[move] = 0
            
        puct_score[move] = branch_score(move)
        
    return prior_dist, visit_dist, avrg_value, puct_score


def plot_stats(prior_dist, visit_dist, avrg_value, puct_score, iteration, out_dir):
    """
    This function will create a plot for the given data. The plots have two
    panels. The top panel shows the average value predicted by the network for
    boards temming from a move and the PUCT score for each move.
    
    The bottom panel shows the prior and visit count distributions over the
    associated moves.
    """
    plt.figure()
    
    barwidth = 0.35
    x = np.arange(len(prior_dist))
    x_ticks = ["({0}, {1}, {2}, {3})".format(*k) for k in prior_dist.keys()]
    
    plt.subplot(212)
    plt.bar(x, prior_dist.values(), barwidth, color='b', label='prior')
    plt.bar(x + barwidth, visit_dist.values(), barwidth, color='g', label='visit')
    
    plt.xticks(x, labels=x_ticks, rotation=90, fontsize=5)
    # plt.yticks([0.0, 0.5, 1.0])
    plt.xlabel('Moves')
    plt.ylabel('Probability')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    
    plt.subplot(211)
    plt.bar(x, avrg_value.values(), barwidth, color='b', label='value')
    plt.bar(x + barwidth, puct_score.values(), barwidth, color='g', label='score')
    plt.ylabel('Score, Value')
    plt.xticks([])
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.title('Iteration {0}'.format(iteration))
    
    plt.savefig(out_dir+'{0}.png'.format(iteration), 
                dpi=300, 
                bbox_inches='tight')
 
    
    
# %% Set up a board position to work with.
player = -1
board_pos =[[ 0, 0, 0,-1, 0, 0, 0],
            [ 0, 0, 1,-1, 0, 0, 0],
            [ 0, 0, 0, 1, 0, 0, 0],
            [ 0, 0,-1, 2, 0,-1,-1],
            [ 0,-1, 0, 1, 0, 0, 0],
            [ 0, 0, 0,-1, 1, 0, 0],
            [ 0, 0, 0,-1, 0, 0, 0]]
board = {}
for i in range(7):
    for j in range(7):
        board[(i, j)] = board_pos[i][j]
game_set = GameSet()
game_set.set_board(board)
game_state = GameState(game_set, player)



# %% Set up the ZeroBot
net = ZeroNet()
bot = ZeroBot(evals_per_turn=21, batch_size=21, network=net)
bot.compile_network((1.0, 1.0))
    


# %% Get the bot to gradually build a search tree and pull out the search stats
# for the legal moves as it does and visualise them.
moves = game_state.legal_moves()
bot.root = bot.create_root_node(game_state)
stats = [get_states(bot.root, moves)]

for i in range(35):
    bot.select_move(game_state)
    stats.append(get_states(bot.root, moves))

out_dir = "zero_search_stats/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for (p, v, a, s), i in zip(stats, range(len(stats))):
    plot_stats(p, v, a, s, i, out_dir)
