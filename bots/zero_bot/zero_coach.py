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

from brandubh_zero import ZeroBot
from zero_training_utils import gain_experience
from zero_training_utils import save_training_data, load_training_data



# %% .
from my_first_zerobot_network import FirstZeroNet

net = FirstZeroNet()
bot = ZeroBot(10, net)



# %% .
from zero_network import ZeroNet

net = ZeroNet()
bot = ZeroBot(50, net)



# %% .
from dual_network import DualNet

net = DualNet()
bot = ZeroBot(5, net)



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



# %%
bot.network.black_model.compile(optimizer=keras.optimizers.Adam(lr=0.0000005),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 1.0])

bot.network.white_model.compile(optimizer=keras.optimizers.Adam(lr=0.0000005),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 1.0])



# %% Evaluate the bot
num_games = 100
bot.evaluate_against_old_bot(num_games)
bot.evaluate_against_rand_bot(num_games)



# %% train the bot
import os

num_episodes = 1
num_cycles = 3

max_num_white_pieces = 1
max_num_black_pieces = 3
moves_limit = 100

for cycle in range(num_cycles):
    
    print('\nEvaluating the bot')
    if (cycle)%1 == 0:
        bot.evaluate_against_old_bot(1)
        bot.evaluate_against_rand_bot(1)
    
    print('\nGainning experience, cycle {0}'.format(cycle))
    experience = gain_experience(bot, num_episodes,
                                 None, 
                                 None,
                                 moves_limit)
    
    print('Preparing training data')
    # Add the generated experience to the bank of training data and load all
    # training data
    training_data = bot.network.create_training_data(experience)
    save_training_data(training_data, cycle)
    training_data = load_training_data()
    
    print('\nTraining network, cycle {0}'.format(cycle))
    losses = bot.network.train(training_data, batch_size=256, epochs=1)
    bot.network.save_losses(losses)
    bot.save_bot("model_data/model_{0}_data/".format(cycle))
        
    
    
    
# %% plot history
import matplotlib.pyplot as plt

score_old = [evaluation[0] for evaluation in bot.evaluation_history_old]
score_ran = [evaluation[0] for evaluation in bot.evaluation_history_ran]

old_err = [((1-evaluation[0]**2)/evaluation[3])**0.5 
           for evaluation in bot.evaluation_history_old]

ran_err = [((1-evaluation[0]**2)/evaluation[3])**0.5 
           for evaluation in bot.evaluation_history_ran]

num_games = [10*evaluation[4] for evaluation in bot.evaluation_history_old]

plt.errorbar(num_games, score_old, old_err, 
             label="Initial zero bot")
plt.errorbar(num_games, score_ran, ran_err,
             label="random bot")
plt.xlabel("Number of self-play games played")
plt.ylabel("Average score against opponent")
plt.title("Zero bot performance history")
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
