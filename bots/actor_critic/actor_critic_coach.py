#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:02:53 2020

@author: john
"""

import numpy as np
import keras

from brandubh import GameState
from actor_critic_bot import ActorCriticBot

def simulate_game(actor_critic_bot, eps):
    
    game = GameState.new_game()
    boards, moves, values, players = [], [], [], []
    encoder = bot.encoder
    
    while game.is_not_over():
        
        if np.random.random() < eps:
            action, value = actor_critic_bot.select_move(game)
            action = actor_critic_bot.rand_bot.select_move(game)
        else:
            action, value = actor_critic_bot.select_move(game)
            
        if action.is_play:
            board_tensor = encoder.encode(game)
            boards.append(board_tensor)
        
            pieces = np.sum(board_tensor[:,:,:2],-1).reshape((7,7))
            moves.append(encoder.encode_move(pieces, action.move))
            
            values.append(value)
            players.append(game.player)
            
        # Make the move.
        game.take_turn_with_no_checks(action)
                
    return boards, moves, values, players, game.winner


def gain_experience(bot, num_episodes, eps=0):
    
    experience = []
    
    for i in range(num_episodes):
        # print('\rrunning episode {0}'.format(i),end='')
        episode = {'boards': [],
                  'moves': [],
                  'values': [],
                  'players': [],
                  'winner': 0}
        
        boards, moves, values, players, winner = simulate_game(bot, eps)
        episode['boards'] = boards
        episode['moves'] = moves
        episode['values'] = values
        episode['players'] = players
        episode['winner'] = winner
        
        experience.append(episode)
       
    # print(' done')
    return experience


def gain_winning_experience(bot, num_episodes, eps=0):
    
    experience = []
    
    for i in range(num_episodes):
        # print('\rrunning episode {0}'.format(i),end='')
        episode = {'boards': [],
                  'moves': [],
                  'values': [],
                  'players': [],
                  'winner': 0}
        
        boards, moves, values, players, winner = simulate_game(bot, eps)
        
        position_of_winning_moves = [i for i, player in enumerate(players)
                                     if player == winner]
        
        episode['boards'] = [boards[i] for i in position_of_winning_moves]
        episode['moves'] = [moves[i] for i in position_of_winning_moves]
        episode['values'] = [values[i] for i in position_of_winning_moves]
        episode['players'] = [players[i] for i in position_of_winning_moves]
        episode['winner'] = winner
        
        experience.append(episode)
       
    # print(' done')
    return experience


def create_training_data(bot, experience):
    # create training data
    X, Y, rewards, advantages = [], [], [], []
    for episode in experience:
        X.append( np.array(episode['boards']) )
        
        # Need to convert the list of move indices into one hot vectors
        # Should use Keras to do this
        moves = episode['moves']
        num_moves = len(moves)
        move_one_hot_vectors = np.zeros((num_moves, 96))
        
        values = episode['values']
        reward = episode['winner'] * np.array(episode['players'])
        advantage = []
        
        for i in range(num_moves):
            move_one_hot_vectors[i, moves[i]] = 1
            advantage.append( reward[i] - values[i] )
        
        Y.append( move_one_hot_vectors )
        rewards.append( reward )
        advantages.append(advantage)
        
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    rewards = np.concatenate(rewards*8)
    advantages = np.concatenate(advantages*8)
    
    X, Y = bot.encoder.expand_data(X, Y)
    
    # samples = X.shape[0]
    
    for i in range(len(advantages)):
        Y[i,:] *= advantages[i]
        
    return X, Y, rewards

# %%
model = ActorCriticBot().init_model()
# %%
bot = ActorCriticBot(model)
# %%
bot.model.compile(optimizer=keras.optimizers.SGD(lr=0.0000005, 
                                            momentum=0.9,
                                            nesterov=True,
                                            clipnorm=1.0),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[1.0, 1.0])

# %%
bot.model.compile(optimizer=keras.optimizers.Adagrad(lr=0.1),
                  loss=['categorical_crossentropy', 'mse'],
                  loss_weights=[0.1, 1.0])

# %% train the bot
import time

num_episodes = 50
num_cycles = 60
epsilon = 0.07

start_time = time.time()
for ii in range(0,3):
    for cycle in range(num_cycles):
        
        print('\nGainning experience, cycle {0}'.format(cycle))
        print('epsilon is {0}'.format(epsilon))
        experience = gain_winning_experience(bot, num_episodes, eps=epsilon)
        
        print('Preparing training data')
        X, Y, rewards = create_training_data(bot, experience)
        
        print('\nTraining network, cycle {0}'.format(cycle))
        bot.model.fit(X, [Y, rewards], batch_size=1500, epochs=2)
        
    epsilon *= 0.9
    
time_taken = time.time() - start_time



# %% Evaluate the bot
num_games = 1000
bot.evaluate_against_old_bot(num_games)
bot.evaluate_against_rand_bot(num_games)