#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 21:43:43 2020

@author: john
"""

import numpy as np
import keras

from brandubh import Act
from four_plane_encoder import FourPlaneEncoder
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.layers import ZeroPadding2D, concatenate



class QBot():
    
    def __init__(self, model=None, eps=0):
        if model:
            self.model = model
        else:
            self.ini_random_model()
            
        self.eps = eps
        self.encoder = FourPlaneEncoder()
        
    def select_move(self, game_state):
        
        board_input = self.encoder.encode(game_state)
        board_input = board_input.reshape(1,7,7,4)
        
        moves = game_state.legal_moves()
        if not moves:
            return Act.pass_turn()
        
        pieces = np.sum(board_input[0,:,:,0:2],-1).reshape((7,7))
        
        move_idxs = [self.encoder.encode_move(pieces, move) for move in moves]
        
        I = np.eye(96)
        Q_values = []
        
        for move in move_idxs:
            move_input = I[move].reshape(1,96)
            Q_values.append(self.model.predict([board_input, move_input]))
            
        best_move = np.argmax(Q_values)
        
        return Act.play(moves[best_move])
    
    def ini_random_model(self):
        board_input = Input(shape=(7,7,4), name='board_input')
        action_input = Input(shape=(96,), name='action_input')
        
        
        conv1a = ZeroPadding2D((2, 2))(board_input)
        conv1b = Conv2D(64, (5, 5), activation='relu')(conv1a)
        conv2a = ZeroPadding2D((1, 1))(conv1b)
        conv2b = Conv2D(64, (3, 3), activation='relu')(conv2a)
        
        flat = Flatten()(conv2b)
        processed_board = Dense(512)(flat)
        
        board_and_action = concatenate([action_input, processed_board])
        hidden_layer = Dense(256, activation='relu')(board_and_action)
        value_output = Dense(1, activation='tanh')(hidden_layer)
        
        model = Model(inputs=[board_input, action_input],
        outputs=value_output)
        
        model.compile(loss='mse',
                      optimizer=keras.optimizers.SGD(lr=0.001, 
                                                    momentum=0.9,
                                                    nesterov=True,
                                                    clipnorm=1.0))
        self.model = model