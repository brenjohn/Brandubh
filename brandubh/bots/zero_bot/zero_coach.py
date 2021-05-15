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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

from brandubh_zero import ZeroBot
from zero_training_utils import gain_experience
from zero_training_utils import save_training_data, load_training_data



# %% .
# from my_first_zerobot_network import FirstZeroNet

# net = FirstZeroNet()
# bot = ZeroBot(10, net)

# bot.network.model.compile(optimizer=keras.optimizers.Adam(lr=0.0000005),
#                   loss=['categorical_crossentropy', 'mse'],
#                   loss_weights=[1.0, 1.0])



# %% .
from networks.zero_network import ZeroNet

net = ZeroNet()
bot = ZeroBot(70, net)

bot.network.model.compile(optimizer=keras.optimizers.Adam(lr=0.000002),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 0.1])



# %% .
from networks.dual_network import DualNet

net = DualNet()
bot = ZeroBot(210, net)

bot.network.black_model.compile(optimizer=keras.optimizers.Adam(lr=0.0000005),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 1.0])

bot.network.white_model.compile(optimizer=keras.optimizers.Adam(lr=0.0000005),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 1.0])



# %% Save the current bot as the old bot to evaluate future versions against.
bot.save_as_old_bot()



# %% train the bot
import os

num_episodes = 21
num_cycles = 280

moves_limit = 100
moves_to_look_ahead = 1

for cycle in range(num_cycles):
    
    if (cycle)%14 == 0:
        print('\nEvaluating the bot')
        alpha_tmp = bot.alpha
        bot.alpha = 0
        bot.evaluate_against_old_bot(350, moves_to_look_ahead)
        bot.evaluate_against_rand_bot(350, moves_to_look_ahead)
        bot.alpha = alpha_tmp
    
    print('\nGainning experience, cycle {0}'.format(cycle))
    eps = 1.0/(1.0 + cycle/14.0) # parameter for epsilon greedy move selection.
    experience = gain_experience(bot, num_episodes,
                                 moves_limit,
                                 eps)
    
    print('Preparing training data')
    # Add the generated experience to the bank of training data and load all
    # training data
    training_data = bot.network.create_training_data(experience)
    save_training_data(training_data, cycle)
    
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

num_games = [evaluation[4] for evaluation in bot.evaluation_history_old]

plt.errorbar(num_games, score_old, old_err, 
             label="Initial zero bot")

plt.errorbar(num_games, score_ran, ran_err,
             label="random bot")

plt.xlabel("Number of epochs. (1 epoch = {0} games)".format(num_episodes))
plt.ylabel("Average score against opponent")
plt.title("ZeroBot performance history")
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

# %%
previous_bot = ZeroBot(1, net)
moves_to_look_ahead = 1
num_games = 350

scores = []
evaluations = []

for i in range(280):
    previous_bot.load_bot("model_data/model_{0}_data/".format(i))
    previous_bot.alpha = 0.0
    previous_bot.evaluate_against_rand_bot(num_games, moves_to_look_ahead)
    scores.append(previous_bot.evaluation_history_ran[-1][0])
    evaluations.append(previous_bot.evaluation_history_ran[-1])
    
# %%
N = 14
averaged_score = [sum([scores[i+j] for j in range(N)])/N for i in range(len(scores)-N)]

fig = plt.figure()
plt.plot(scores, label="Ave. score from {0} games".format(num_games),
         color='grey',
         linewidth=0.7)
plt.plot(range(N//2), len(scores)-N//2, averaged_score,
         label='Moving average N = {0}'.format(N),
         linewidth=4)
plt.title("ZeroBot evaluation against RandBot with 0 look-a-head")
plt.xlabel("Number of cycles")
plt.ylabel("Score")
plt.legend()

fig.savefig('ZeroBot_moving_ave_N14_evaluation.png', dpi=300)
