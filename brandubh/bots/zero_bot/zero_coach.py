#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:18:13 2020

@author: john

This file is divided into cells recognised by spyder. Each cell provides a
block of code which may be useful for training a ZeroBot.
"""
import sys
sys.path.append("..")
sys.path.append("../..")

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
# from networks.zero_network import print_tensor

from brandubh_zero import ZeroBot
from greedy_random_bot import GreedyRandomBot
from training_utils import gain_experience, save_training_data, DataManager, competition




# %% .
from networks.zero_network import ZeroNet

net = ZeroNet()
bot = ZeroBot(evals_per_turn=70, batch_size=28, network=net)

def compile_bot(bot):
    bot.network.model.compile(optimizer=keras.optimizers.Adam(),
                              loss=['categorical_crossentropy', 'mse'],
                              loss_weights=[1.0, 0.1])
    
compile_bot(bot)



# %% Save the current bot as the old bot to evaluate future versions against.
bot.save_as_old_bot()



# %% train the bot
import os

num_episodes = 4
num_cycles = 1

move_limit = 140
moves_to_look_ahead = 70

dm = DataManager()

for cycle in range(num_cycles):
    
    print('\nEvaluating the bot')
    alpha_tmp = bot.alpha
    bot.alpha = 0
    bot.evaluate_against_rand_bot(7, moves_to_look_ahead)
    bot.evaluate_against_grnd_bot(7, moves_to_look_ahead)
    # if (cycle)%7 == 0:
        # bot.evaluate_against_mcts_bot(35, moves_to_look_ahead)
    bot.alpha = alpha_tmp
    
    print('\nGainning experience, cycle {0}'.format(cycle))
    # eps = 1.0/(1.0 + (cycle+7)/0.7) + 0.035# parameter for epsilon greedy selection.
    # eps = 0.07
    eps = 0.07
    self_play_exp = gain_experience(bot, bot, num_episodes, move_limit, eps)
    # grand_wht_exp = gain_experience(gr_bot, bot, num_episodes, move_limit, 0)
    # grand_blk_exp = gain_experience(bot, gr_bot, num_episodes, move_limit, 0)
    # experience = self_play_exp + grand_wht_exp + grand_blk_exp
    # experience = grand_wht_exp + grand_blk_exp
    experience = self_play_exp
    # experience = grand_wht_exp
    
    print('Preparing training data')
    # Add the generated experience to the bank of training data and load all
    # training data
    training_data = bot.network.create_training_data(experience)
    save_training_data(training_data, cycle)
    dm.append_data(training_data)
    
    print('\nTraining network, cycle {0}'.format(cycle))
    n = cycle + 1 if cycle < 7 else 7
    # n = 1
    for i in range(n):
        training_data = dm.sample_training_data(4096)
        losses = bot.network.train(training_data, batch_size=256)
    
    # lr = 0.00001/(1.0 + (cycle+1)/0.7)
    # compile_bot(bot)
    # bot.save_bot("model_data/model_{0}_data/".format(cycle))
    
# %%
import matplotlib.pyplot as plt

score_rand             = [s[0] for s in bot.evaluation_history_rand]
self_games_played_rand = [s[4] for s in bot.evaluation_history_rand]

score_grnd             = [s[0] for s in bot.evaluation_history_grnd]
self_games_played_grnd = [s[4] for s in bot.evaluation_history_grnd]

score_mcts             = [s[0] for s in bot.evaluation_history_mcts]
self_games_played_mcts = [s[4] for s in bot.evaluation_history_mcts]

plt.plot(self_games_played_rand, score_rand, label="random bot")
plt.plot(self_games_played_grnd, score_grnd, label="greedy random bot")
plt.plot(self_games_played_mcts, score_mcts, label="MCTS bot")

plt.legend()
