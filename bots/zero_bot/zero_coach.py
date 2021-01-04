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

import keras
import copy

from brandubh_zero import ZeroBot
from zero_training_utils import gain_experience, create_training_data



# %% .
from my_first_zerobot_network import FirstZeroNet

net = FirstZeroNet()
bot = ZeroBot(10, net)



# %% .
from zero_network import ZeroNet

net = ZeroNet()
bot = ZeroBot(50, net)



# %% Save the current bot as the old bot to evaluate future versions against.
bot.save_as_old_bot()



# %%
bot.model.compile(optimizer=keras.optimizers.SGD(lr=0.0000001, 
                                            momentum=0.9,
                                            nesterov=True,
                                            clipnorm=1.0),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[0.5, 1.0])



# %%
bot.network.model.compile(optimizer=keras.optimizers.Adam(lr=0.0000005),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 1.0])



# %% Evaluate the bot
num_games = 100
bot.evaluate_against_old_bot(num_games)
bot.evaluate_against_rand_bot(num_games)



# %% train the bot
import os

num_episodes = 50
num_cycles = 2000

max_num_white_pieces = 4
max_num_black_pieces = 8
moves_limit = 500

model_num = 0

for cycle in range(num_cycles):
    
    if (cycle)%10 == 0:
        bot.evaluate_against_old_bot(100)
        bot.evaluate_against_rand_bot(100)
        model_directory = "model_{0}_data/".format(model_num)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        bot.save_bot(model_directory)
        model_num += 1
    
    print('\nGainning experience, cycle {0}'.format(cycle))
    experience = gain_experience(bot, num_episodes,
                                 max_num_white_pieces, 
                                 max_num_black_pieces,
                                 moves_limit)
    
    print('Preparing training data')
    X, Y, rewards = create_training_data(bot, experience)
    
    print('\nTraining network, cycle {0}'.format(cycle))
    print('Training model on {0} moves'.format(len(X)))
    losses = bot.network.model.fit(X, [Y, rewards], batch_size=128, epochs=1)
    bot.save_losses(losses)
        
    
    
    
# %% plot history
import matplotlib.pyplot as plt

score_old = [evaluation[0] for evaluation in bot.evaluation_history_old]
score_ran = [evaluation[0] for evaluation in bot.evaluation_history_ran]

old_err = [((1-evaluation[0]**2)/evaluation[3])**0.5 
           for evaluation in bot.evaluation_history_old]

ran_err = [((1-evaluation[0]**2)/evaluation[3])**0.5 
           for evaluation in bot.evaluation_history_ran]

plt.errorbar(range(len(bot.evaluation_history_old)), score_old, old_err, 
             label="old")
plt.errorbar(range(len(bot.evaluation_history_ran)), score_ran, ran_err,
             label="random")
plt.legend()



# %% print history
# evaluation history = [score, wins as white, wins as black, number of games]
print("old bot:")
print(bot.evaluation_history_old)
print("random bot:")
print(bot.evaluation_history_ran)



# %%
bot.save_bot()

# %%
bot.load_bot()
