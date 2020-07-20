#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:53:45 2020

@author: john
"""
import numpy as np
import keras
import random

from brandubh import GameState
from random_bot import RandomBot
from dlbot import DeepLearningBot
from four_plane_encoder import FourPlaneEncoder



def simulate_game(black_bot, white_bot, encoder):
    
    game = GameState.new_game()
    boards, moves, players = [], [], []
    
    while game.is_not_over():
        
        if game.player == 1:
            action = white_bot.select_move(game)
        else:
            action = black_bot.select_move(game)
            
        if action.is_play:
            board_tensor = encoder.encode(game)
            boards.append(board_tensor)
        
            pieces = np.sum(board_tensor[:,:,:2],-1).reshape((7,7))
            moves.append(encoder.encode_move(pieces, action.move))
            
            players.append(game.player)
            
        # Make the move.
        game.take_turn_with_no_checks(action)
                
    return boards, moves, players, game.winner



def gain_experience(bot, encoder, num_episodes):
    
    experience = []
    
    for i in range(num_episodes):
        # print('\rrunning episode {0}'.format(i),end='')
        episode = {'boards': [],
                  'moves': [],
                  'players': [],
                  'winner': 0}
        
        boards, moves, players, winner = simulate_game(bot, bot, encoder)
        episode['boards'] = boards
        episode['moves'] = moves
        episode['players'] = players
        episode['winner'] = winner
        
        experience.append(episode)
       
    # print(' done')
    return experience



def gain_experience_against_rand_bot(bot, encoder, num_episodes):
    
    experience = []
    rbot = RandomBot()
    
    for i in range(num_episodes):
        # print('\rrunning episode {0}'.format(i),end='')
        episode = {'boards': [],
                  'moves': [],
                  'players': [],
                  'winner': 0}
        
        if i%2==0:
            boards, moves, players, winner = simulate_game(bot, rbot, encoder)
            for i, player in enumerate(players):
                if player==1:
                    del boards[i]
                    del moves[i]
                    del players[i]
        else:
            boards, moves, players, winner = simulate_game(rbot, bot, encoder)
            for i, player in enumerate(players):
                if player==1:
                    del boards[i]
                    del moves[i]
                    del players[i]
                    
        episode['boards'] = boards
        episode['moves'] = moves
        episode['players'] = players
        episode['winner'] = winner
        
        experience.append(episode)
       
    # print(' done')
    return experience



def evaluate_against_rand_bot(model, num_games):
    bot_rd = RandomBot()
    bot_nn = DeepLearningBot(model)
    player_nn = 1
    score = 0
    
    for i in range(num_games):
        # print('\rplaying game {0}'.format(i), end='')
        game = GameState.new_game()
        
        while game.is_not_over():
            if game.player == player_nn:
                action = bot_nn.select_move(game)
            else:
                action = bot_rd.select_move(game)
                
            game.take_turn_with_no_checks(action)
                
        score += player_nn*game.winner
        player_nn *= -1
        
    # print(' done')
    return score/num_games
    
        

# %%
from keras.models import load_model
model = load_model('dl_model.h5')

# %% Save model
model.save('dl_model_temp.h5')
        

# %%
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.1, clipnorm=1.0))

agent = DeepLearningBot()
encoder = FourPlaneEncoder()
num_episodes = 100

experience = gain_experience(agent, encoder, num_episodes)


# %%
X, Y, reward = [], [], []
for episode in experience:
    X.append( np.array(episode['boards']) )
    Y.append( np.array(episode['moves']) )
    reward.append( episode['winner'] * np.array(episode['players']) )
    
X = np.concatenate(X)
Y = np.concatenate(Y)
reward = np.concatenate(reward*8)

encoder = FourPlaneEncoder()

X, Y = encoder.expand_data(X, Y)

# %%
from keras.models import load_model
# model = load_model('dl_model_temp.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.00001,
                                             clipnorm=1.0))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.001, 
                                            momentum=0.9,
                                            nesterov=True,
                                            clipnorm=1.0))


# %%
from keras.models import load_model
model = load_model('dl_model_temp.h5')

# %% Save model
model.save('dl_model_relu.h5')
np.save('score_history_relu.npy', score)

# %%
import time
start_time = time.time()
# score = []
score = np.load('score_history.npy')
score = list(score)

for cycle in range(40):
    
    print('\nGain experience step, cycle {0}'.format(cycle))
    agent = DeepLearningBot(model)
    encoder = FourPlaneEncoder()
    num_episodes = 10

    experience = gain_experience(agent, encoder, num_episodes)
    
    print('Preparing training data')
    # create training data
    X, Y, reward = [], [], []
    for episode in experience:
        X.append( np.array(episode['boards']) )
        
        # Need to convert the list of move indices into one hot vectors
        # Should use Keras to do this
        moves = episode['moves']
        num_moves = len(moves)
        move_one_hot_vectors = np.zeros((num_moves, 96))
        for i in range(num_moves):
            move_one_hot_vectors[i, moves[i]] = 1
        
        Y.append( move_one_hot_vectors )
        reward.append( episode['winner'] * np.array(episode['players']) )
        
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    reward = np.concatenate(reward)
    
    # X, Y = encoder.expand_data(X, Y)
    
    samples = X.shape[0]
    
    for i in range(len(reward)):
        Y[i,:] *= reward[i]
     
    # random_indices = [i for i in range(samples)]
    # random.shuffle(random_indices)
    
    # X = X[random_indices,:,:,:]
    # Y = Y[random_indices,:]
    # X = X.reshape(samples, 7, 7, 4)
    # Y = Y.reshape(samples, 96)
    
    
    
    print('\nTraining step, cycle {0}'.format(cycle))
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=keras.optimizers.SGD(lr=0.00001, clipnorm=1.0))
    
    model.fit(X, Y,
              batch_size=500,
              epochs=5)
    
    
    if (cycle+1)%40 == 0:
        print('\nEvaluation step, cycle {0}'.format(cycle))
        num_games = 200
        score.append( evaluate_against_rand_bot(model, num_games) )
        
        print('\nSaving model\n')
        model.save('dl_model_temp.h5')
        np.save('score_history.npy', score)
        
time_taken = time.time() - start_time
ave_score = [np.sum(score[:i])/i for i in range(1,len(score)+1)]
running_ave = np.convolve(score, np.ones((10,))/10, mode='valid')