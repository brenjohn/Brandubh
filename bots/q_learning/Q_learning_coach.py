#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:50:05 2020

@author: john
"""

from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.layers import ZeroPadding2D, concatenate


# %%
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
