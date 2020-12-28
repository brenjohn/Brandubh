#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:16:53 2020

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import copy

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.layers import LeakyReLU, add
from keras.models import load_model


class ZeroNet():
    
    def __init__(self):
        self.model = ZeroNet.build_model()
        self.model.summary()
        self.encoder = SixPlaneEncoder()
        
    @classmethod
    def conv_layer(cls, x, filters, kernel_size):
        
        x = Conv2D(filters, kernel_size, use_bias = True,
                   padding = 'same', activation = 'linear')(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x
    
    @classmethod
    def residual_layer(cls, input_block, filters, kernel_size):
        
        x = ZeroNet.conv_layer(input_block, filters, kernel_size)
        x = Conv2D(filters, kernel_size, use_bias = True,
                   padding = 'same', activation = 'linear')(x)
        # x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x
    
    @classmethod
    def value_head(cls, x):
        
        x = Conv2D(filters = 32, kernel_size = (3, 3), use_bias = True,
                   padding = 'same', activation = 'linear')(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(64, use_bias = True, 
                  activation = 'linear')(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias = True, activation = 'tanh', 
                  name = 'value_head')(x)
        return x
    
    @classmethod
    def policy_head(cls, x):
        x = Conv2D(filters = 24, kernel_size = (3, 3), use_bias = True,
                   padding = 'same', activation = 'linear')(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters = 24, kernel_size = (3, 3), use_bias = True,
                   padding = 'same', activation = 'softmax',
                   name = 'policy_head')(x)
        return x
    
    @classmethod
    def build_model(cls):
        
        board_input = Input(shape=(7,7,6), name='board_input')
        
        processed_board = ZeroNet.conv_layer(board_input, 64, (3, 3))
        for i in range(7):
            processed_board = ZeroNet.residual_layer(processed_board, 64, (3, 3))
            
        value_output = ZeroNet.value_head(processed_board)
        policy_output = ZeroNet.policy_head(processed_board)
        
        model = Model(inputs=board_input, 
                      outputs=[policy_output, value_output])
        return model
    
    def predict(self, game_state):
        # Should take gamestate as input and output a dictionary of
        # move-prior pairs
        input_tensor, pieces = self.encoder.encode(game_state, True)
        priors, value = self.model.predict(input_tensor.reshape(1, 7, 7, 6))
        move_priors = self.encoder.decode_policy(priors[0], pieces)
        
        # Normalise the move prior distribution
        N = sum(move_priors.values())
        for (move, prior) in move_priors.items():
            move_priors[move] = prior/N
        return move_priors, value[0][0]
    
    def save_network(self, prefix="model_data/"):
        self.model.save(prefix + 'zero_model.h5')
        load_command = "from bots.zero_bot.zero_network import ZeroNet; "
        load_command += "self.network = ZeroNet()"
        return load_command
        
    def load_network(self, prefix="model_data/"):
        self.model = load_model(prefix + 'zero_model.h5')
    
    
class SixPlaneEncoder():
    def __init__(self):
        self.convert_to_tensor_element = {1: 'board_tensor[r,c,0] = 1',
                                          2: 'board_tensor[r,c,0] = 1;' +
                                             'board_tensor[r,c,1] = 1',
                                         -1: 'board_tensor[r,c,2] = 1',
                                         -2: 'board_tensor[r,c,2] = 1;' +
                                             'board_tensor[r,c,3] = 1'}
        
    def encode(self, game_state, return_pieces=False):
        board_tensor = np.zeros((7,7,6))
        player = game_state.player
        
        if player == 1:
            pieces = game_state.game_set.white_pieces
            board_tensor[:, :, 4] = 1
        else:
            pieces = game_state.game_set.black_pieces
            board_tensor[:, :, 5] = 1
        
        for (r, c) in game_state.game_set.white_pieces:
            piece = game_state.game_set.board[(r, c)]
            exec(self.convert_to_tensor_element[piece*player])
            
        for (r, c) in game_state.game_set.black_pieces:
            exec(self.convert_to_tensor_element[-1*player])
        
        if return_pieces:
            return board_tensor, pieces
        return board_tensor
    
    def decode_policy(self, model_output, pieces):
        # should return a dictionary of move - prior pairs
        move_priors = {}
        for (xi, yi) in pieces:
            
            for i, yf in enumerate(range(yi-1, -1, -1)):
                move_priors[(xi, yi, xi, yf)] = model_output[xi, yi, i]
                
            for i, yf in enumerate(range(yi+1, 7)):
                move_priors[(xi, yi, xi, yf)] = model_output[xi, yi, i+6]
                
            for i, xf in enumerate(range(xi-1, -1, -1)):
                move_priors[(xi, yi, xf, yi)] = model_output[xi, yi, i+12]
                
            for i, xf in enumerate(range(xi+1, 7)):
                move_priors[(xi, yi, xf, yi)] = model_output[xi, yi, i+18]
        
        return move_priors
    
    def encode_prior(self, move_probs):
        target_tensor = np.zeros((7,7,24))
        # Normalising constant
        N = 0 
        
        for move, prob in move_probs.items():
            xi, yi, xf, yf = move
            if xi == xf:
                k = yf - yi
                k = k + 6 if k < 0 else k + 5
            else:
                k = xf - xi
                k = k + 18 if k < 0 else k + 17
            target_tensor[xi, yi, k] = prob
            N += prob
        
        return target_tensor/N
    
    def encode_priors(self, priors):            
        encoded_priors = [self.encode_prior(prior) for prior in priors]
        num_moves = len(encoded_priors)
        return np.reshape(encoded_priors, (num_moves, 7, 7, 24))
    
    def expand_data(self, X, Y):
        
        X_temp = copy.copy(X)
        Y_temp = copy.copy(Y)
        
        X_temp = np.transpose(X_temp, (0,2,1,3))
        Y_temp = self.transpose_policy(Y_temp)
        
        X = np.concatenate((X, copy.copy(X_temp)), axis=0)
        Y = np.concatenate((Y, copy.copy(Y_temp)), axis=0)
        
        for i in range(3):
            
            X_temp = np.flip(X_temp, axis=1)
            Y_temp = self.flip_policy(Y_temp)
            
            X = np.concatenate((X, copy.copy(X_temp)), axis=0)
            Y = np.concatenate((Y, copy.copy(Y_temp)), axis=0)
            
            X_temp = np.transpose(X_temp, (0,2,1,3))
            Y_temp = self.transpose_policy(Y_temp)
            
            X = np.concatenate((X, copy.copy(X_temp)), axis=0)
            Y = np.concatenate((Y, copy.copy(Y_temp)), axis=0)
            
        return X, Y
    
    def transpose_policy(self, Y):
        Y = np.transpose(Y, (0, 2, 1, 3))
        Y = np.concatenate([Y[:, :, :, 12:], Y[:, :, :, 0:12]], axis=3)
        return Y
    
    def flip_policy(self, Y):
        Y = np.flip(Y, axis=1)
        Y[:,:,:,12:] = np.concatenate([Y[:, :, :, 18:], Y[:, :, :, 12:18]], 
                                      axis=3)
        return Y