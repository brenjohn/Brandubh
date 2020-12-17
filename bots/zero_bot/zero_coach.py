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
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.layers import Activation, LeakyReLU, add



# %% Build a model similar to the one used by AlphaZero
from keras import regularizers
reg_const = 0.0001

def conv_layer(x, filters, kernel_size):
    
    x = Conv2D(filters, kernel_size, use_bias = False,
               padding = 'same', activation = 'linear',
               kernel_regularizer = regularizers.l2(reg_const))(x)
    # x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    return x

def residual_layer(input_block, filters, kernel_size):
    
    x = conv_layer(input_block, filters, kernel_size)
    x = Conv2D(filters, kernel_size, use_bias = False,
               padding = 'same', activation = 'linear',
               kernel_regularizer = regularizers.l2(reg_const))(x)
    # x = BatchNormalization(axis=1)(x)
    x = add([input_block, x])
    x = LeakyReLU()(x)
    return x

def value_head(x):
    
    x = Conv2D(filters = 1, kernel_size = (1, 1), use_bias = False,
               padding = 'same', activation = 'linear',
               kernel_regularizer = regularizers.l2(reg_const))(x)
    # x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(20, use_bias = False, 
              activation = 'linear',
              kernel_regularizer = regularizers.l2(reg_const))(x)
    x = LeakyReLU()(x)
    x = Dense(1, use_bias = False, activation = 'tanh', name = 'value_head',
              kernel_regularizer = regularizers.l2(reg_const))(x)
    return x

def policy_head(x):
    x = Conv2D(filters = 2, kernel_size = (1, 1), use_bias = False,
               padding = 'same', activation = 'linear',
               kernel_regularizer = regularizers.l2(reg_const))(x)
    # x = BatchNormalization(axis=1)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(96, use_bias = False, 
              activation = 'linear',
              kernel_regularizer = regularizers.l2(reg_const))(x)
    x = LeakyReLU()(x)
    x = Dense(96, use_bias = False, activation = 'softmax', 
              name = 'policy_head',
              kernel_regularizer = regularizers.l2(reg_const))(x)
    return x

def build_zero_model():
    
    board_input = Input(shape=(7,7,4), name='board_input')
    
    processed_board = conv_layer(board_input, 64, (3, 3))
    for i in range(3):
        processed_board = residual_layer(processed_board, 64, (3, 3))
        
    value_output = value_head(processed_board)
    policy_output = policy_head(processed_board)
    
    model = Model(inputs=board_input, 
                  outputs=[policy_output, value_output])
    return model

model = build_zero_model()
model.summary()
bot = ZeroBot(100, model)


# %% Create a neural network for a ZeroBot

board_input = Input(shape=(7,7,4), name='board_input')

processed_board = Conv2D(16, (3, 3), 
                         padding='same',
                         activation='relu')(board_input)
for i in range(4):
    skip = processed_board
    processed_board = Conv2D(16, (3, 3), 
                             padding='same',
                             activation='relu')(processed_board)
    processed_board = add([processed_board, skip])

# policy_conv = Conv2D(2, (1, 1), activation='relu')(processed_board)
policy_flat = Flatten()(processed_board)
policy_hidden = Dense(128, activation='relu')(policy_flat)
policy_output = Dense(96, activation='softmax')(policy_hidden)

# value_conv = Conv2D(1, (1, 1), activation='relu')(processed_board)
value_flat = Flatten()(processed_board)
value_hidden = Dense(64, activation='relu')(value_flat)
value_output = Dense(1, activation='tanh')(value_hidden)

model = Model(inputs=board_input, 
              outputs=[policy_output, value_output])
model.summary()

bot = ZeroBot(36, model)



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
bot.model.compile(optimizer=keras.optimizers.Adam(lr=0.000002),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 1.0])



# %% Evaluate the bot
num_games = 100; num_white_pieces = 4; num_black_pieces = 8
bot.evaluate_against_old_bot(num_games, num_white_pieces, num_black_pieces)
bot.evaluate_against_rand_bot(num_games, num_white_pieces, num_black_pieces)



# %% Evaluate the bot
num_games = 400
bot.evaluate_against_old_bot(num_games)
bot.evaluate_against_rand_bot(num_games)



# %% save the base bot
zero_bot_base = copy.deepcopy(bot)



# %% evaluate the current bot against the base bot
num_games = 10
results = bot.evaluate_against_bot(zero_bot_base, num_games)
# If this is positive then then current bot won more often
print('\n' + str(results[0]))



# %% use the base bot to set the current bot
bot = copy.deepcopy(zero_bot_base)



# %% train the bot
import numpy as np

num_episodes = 1
num_cycles = 200

max_num_black_pieces = 8
max_num_white_pieces = 4
moves_limit = 500

for cycle in range(num_cycles):
    
    print('\nGainning experience, cycle {0}'.format(cycle))
    experience = gain_experience(bot, num_episodes,
                                 max_num_white_pieces, max_num_black_pieces,
                                 moves_limit)
    
    print('Preparing training data')
    X, Y, rewards = create_training_data(bot, experience)
    
    print('\nTraining network, cycle {0}'.format(cycle))
    losses = bot.model.fit(X, [Y, rewards], batch_size=256, epochs=1)
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

# %%
import multiprocessing as mp
import numpy as np
import copy

def generate_training_data(q, bot, num_episodes):
    np.random.seed()
    print('\nGainning experience, cycle {0}'.format(1))
    experience = gain_experience(bot, num_episodes)
    
    print('Preparing training data')
    X, Y, rewards = create_training_data(bot, experience)
    
    q.put((X, Y, rewards))
    
num_episodes = 1
processes = []
q = mp.Queue()

bots = [bot, ZeroBot(10, copy.copy(bot.model))]

for i in range(2):
    p = mp.Process(target=generate_training_data, args=(q, bots[i], num_episodes))
    processes.append(p)
    p.start()
    
for process in processes:
    process.join()
    
X, Y, rewards = q.get()
for _ in range(1):
    Xi, Yi, ri = q.get()
    X = np.concatenate((X, Xi), axis=0)
    Y = np.concatenate((Y, Yi), axis=0)
    rewards = np.concatenate((rewards, ri), axis=0)
    
print('\nTraining network, cycle {0}'.format(cycle))
bot.model.fit(X, [Y, rewards], batch_size=128, epochs=1)

# %%
num_games = 100; num_white_pieces = 0; num_black_pieces = 7
bot.evaluate_against_old_bot(num_games, num_white_pieces, num_black_pieces)
bot.evaluate_against_rand_bot(num_games, num_white_pieces, num_black_pieces)

num_episodes = 10
num_cycles = 10
num_white_pieces = 0; num_black_pieces = 8

for cycle in range(num_cycles):
    print('\nGainning experience, cycle {0}'.format(cycle))
    experience = gain_experience(bot, num_episodes, 
                                 num_white_pieces, num_black_pieces)
    
    print('Preparing training data')
    X, Y, rewards = create_training_data(bot, experience)
    
    print('\nTraining network, cycle {0}'.format(cycle))
    losses = bot.model.fit(X, [Y, rewards], batch_size=128, epochs=1)
    bot.save_losses(losses)