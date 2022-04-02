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
from networks.zero_network import print_tensor

from brandubh_zero import ZeroBot
from greedy_random_bot import GreedyRandomBot
from training_utils import gain_experience, save_training_data, DataManager, competition



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
bot = ZeroBot(evals_per_turn=700, batch_size=21, network=net)

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
num_cycles = 420

move_limit = 140
moves_to_look_ahead = 1

# gr_bot = GreedyRandomBot()

dm = DataManager()

for cycle in range(num_cycles):
    
    if (cycle)%1 == 0:
        print('\nEvaluating the bot')
        alpha_tmp = bot.alpha
        bot.alpha = 0
        #bot.evaluate_against_old_bot(350, moves_to_look_ahead)
        bot.evaluate_against_rand_bot(140, moves_to_look_ahead)
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
        
    
    
    
# %% plot
import matplotlib.pyplot as plt
import numpy as np

title = "4-2-2022-a"

score_hist = [sc[0] for sc in bot.evaluation_history_ran]
moving_av = np.convolve(score_hist, np.ones(14), 'valid')/14
plt.plot(score_hist)
plt.plot(range(6, len(score_hist)-7), moving_av)
plt.title(title)
plt.xlabel('iterations')
plt.ylabel('average score')

plt.savefig('evaluation-'+title+'.png', dpi=300)

# %% .
from networks.zero_network import ZeroNet

def compile_bot(bot, lr):
    bot.network.model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                              loss=['categorical_crossentropy', 'mse'],
                              loss_weights=[1.0, 0.1])
    
net = ZeroNet()
bot_a = ZeroBot(350, net)
compile_bot(bot_a, 0.000001)

net = ZeroNet()
bot_b = ZeroBot(350, net)
compile_bot(bot_b, 0.000001)

print('Evaulating student and teacher bots')
# bot_a.turn_off_look_a_head(); bot_b.turn_off_look_a_head()
winner = competition(bot_a, bot_b, num_games=100)
# bot_a.turn_on_look_a_head(); bot_b.turn_on_look_a_head()
if winner is bot_a:
    teacher_bot, student_bot = bot_a, bot_b
else:
    teacher_bot, student_bot = bot_b, bot_a
    
    
student_bot.save_bot("model_data/student_model_{0}_data/".format(-1))
teacher_bot.save_bot("model_data/teacher_model_{0}_data/".format(-1))

teacher_bot.evaluate_against_rand_bot(350, 1)


# %% train the bot
import os

num_episodes = 70
num_cycles = 70

move_limit = 140


for cycle in range(num_cycles):
    
    print('\nGainning experience, cycle {0}'.format(cycle))
    eps = 1.0/(1.0 + (cycle+14)/7.0) # parameter for epsilon greedy selection.
    experience = gain_experience(teacher_bot, teacher_bot, num_episodes, move_limit, 0)
    
    print('Preparing training data')
    # Add the generated experience to the bank of training data and load all
    # training data
    training_data = student_bot.network.create_training_data(experience)
    # save_training_data(training_data, cycle)
    
    print('\nTraining network, cycle {0}'.format(cycle))
    losses = student_bot.network.train(training_data, batch_size=256, epochs=2)
    student_bot.network.save_losses(losses)
    
    # lr = 0.00001/(1.0 + (cycle+1)/0.7)
    # compile_bot(bot, 0.000002)
    student_bot.save_bot("model_data/student_model_{0}_data/".format(cycle))
    
    print('Evaluating student and teacher bots')
    # student_bot.turn_off_look_a_head(); teacher_bot.turn_off_look_a_head()
    winner = competition(student_bot, teacher_bot, num_games=100, moves_limit=140, threshold=10)
    # student_bot.turn_on_look_a_head(); teacher_bot.turn_on_look_a_head()
    if winner is student_bot:
        print('And the student becomes the teacher')
        teacher_bot, student_bot = student_bot, teacher_bot
        teacher_bot.save_bot("model_data/teacher_model_{0}_data/".format(cycle))
        teacher_bot.evaluate_against_rand_bot(350, 1)
    
    
    
# %% plot history
import matplotlib.pyplot as plt

# score_old = [evaluation[0] for evaluation in bot.evaluation_history_old]
score_ran = [evaluation[0] for evaluation in bot.evaluation_history_ran]

# old_err = [((1-evaluation[0]**2)/evaluation[3])**0.5 
#            for evaluation in bot.evaluation_history_old]

ran_err = [((1-evaluation[0]**2)/evaluation[3])**0.5 
           for evaluation in bot.evaluation_history_ran]

num_games = [evaluation[4] for evaluation in bot.evaluation_history_ran]

# plt.errorbar(num_games, score_old, old_err, 
#              label="Initial zero bot")

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
# previous_bot = ZeroBot(1, net)
previous_bot = student_bot
moves_to_look_ahead = 1
num_games = 350

scores = []
evaluations = []

for i in range(-1, 15):
    previous_bot.load_bot("model_data/student_model_{0}_data/".format(i))
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
