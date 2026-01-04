#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:16:53 2020

@author: john

This file defines two classes which define a neural network and implements the
interface a ZeroBot needs to interact with the neural network in order to 
predict the value of board positions, predict valuable moves that can be made
from a board position and to prepare training data to train and improve the
network.

The ZeroNet class has a keras neural network for predicting the value of a
board-position/game-state and the distribution of visits a ZeroBot will make
to branches of the decision tree stemming from the board-position.

The SixPlaneEncoder class is used to convert a game-state to an input tensor
for the neural network and to convert the output of the network to a 
dictionary of move-value pairs. It also has methods for expanding training 
data for the network.
"""
# Disable tensorflow logging messages:
import logging
import os
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Normal imports:
import numpy as np

from .encoder import SixPlaneEncoder

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.layers import LeakyReLU, add, Softmax, Reshape
from keras.models import load_model
from keras.regularizers import l2
from keras.optimizers import Adam

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ZeroNet():
    """
    This class has methods for building, saving and loading a nerual network
    to be used by a Zerobot.
    
    The network architecture is as follows:
        The input layer consists of 6 7x7 arrays of neurons (see encoder class)
        
        The input is then passed to a convolutional layer with 64 filters and
        a 3x3 kernal
        
        This is followed by 7 residual layers which consist of two consecutive
        convolutional layers with 64 filters each, produced by 3x3 kernals,
        and a skip connection connecting the input of the first convolutional
        layer to the output of the second. Leaky ReLu functions are used
        as activations for each convolutional layer.
        
        The output of the last residual layer is then passed to two different
        output heads - a value head and a policy head.
        
        The value head consists of a single convolution layer with 32 filters
        connected to a dense hidden layer of 64 nodes followed by a single
        output node. Leaky ReLus are used as activations for each layer expect 
        the output node which uses a tanh as an activation.
        
        The policy head has two consecutive convolutional layers with 24 
        filters. The first uses a leaky ReLu activation and the second uses a
        softmax activation. The output of the policy head is a 7x7x24 tensor
        (see Encoder class)
    """
    alpha = 0.0001
    biases = True
    
    def __init__(self):
        self.model = ZeroNet.build_model()
        self.compile_lite_model()
        self.encoder = SixPlaneEncoder()
        
    def compile_lite_model(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        lite_model = converter.convert()
        
        self.intrp = tf.lite.Interpreter(model_content=lite_model)
        self.intrp.allocate_tensors()
        
        input_det = self.intrp.get_input_details()[0]
        policy_det, value_det = self.intrp.get_output_details()
        
        self.inp_ind = input_det["index"]
        self.val_ind = value_det["index"]
        self.pol_ind = policy_det["index"]
        
    @classmethod
    def conv_layer(cls, x, filters, kernel_size):
        
        x = Conv2D(filters, kernel_size, use_bias = cls.biases,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x
    
    @classmethod
    def residual_layer(cls, input_block, filters, kernel_size):
        x = ZeroNet.conv_layer(input_block, filters, kernel_size)
        x = Conv2D(filters, kernel_size, use_bias = cls.biases,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        # x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x
    
    @classmethod
    def value_head(cls, x):
        x = Conv2D(filters = 35, kernel_size = (1, 1), use_bias = cls.biases,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(21, use_bias = cls.biases, 
                  activation = 'linear',
                  bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        x = LeakyReLU()(x)
        x = Dense(14, use_bias = cls.biases, 
                  activation = 'linear',
                  bias_regularizer = l2(cls.alpha),
                  kernel_regularizer = l2(cls.alpha))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias = cls.biases, activation = 'tanh', 
                  name = 'value_head')(x)
        return x
    
    @classmethod
    def policy_head(cls, x):
        x = Conv2D(filters = 35, kernel_size = (1, 1), use_bias = cls.biases,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        x = LeakyReLU()(x)
        
        x = Conv2D(filters = 24, kernel_size = (1, 1), use_bias = cls.biases,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        # x = LeakyReLU()(x)
        
        x = Reshape(target_shape = (-1,))(x)
        x = Softmax()(x)
        x = Reshape(target_shape = (7, 7, 24), name='policy_head')(x)
        return x
    
    @classmethod
    def build_model(cls):
        """
        This method builds the network model when a ZeroNet is initialised.
        """
        board_input = Input(shape=(7,7,6), name='board_input')
        
        processed_board = ZeroNet.conv_layer(board_input, 35, (3, 3))
        for i in range(7):
            processed_board = ZeroNet.residual_layer(processed_board, 35, (3, 3))
            
        policy_output = ZeroNet.policy_head(processed_board)
        value_output = ZeroNet.value_head(processed_board)
        
        return Model(
            inputs=board_input,
            outputs=[policy_output, value_output]
        )
    
    def compile_network(
            self, 
            policy_weight, 
            value_weight,
            learning_rate = 0.0001
        ):
        self.model.compile(
            optimizer = Adam(learning_rate=learning_rate,),
            loss = ['categorical_crossentropy', 'mse'],
            loss_weights = [policy_weight, value_weight]
        )
    
    def predict(self, game_states):
        """
        This method uses the neural network to predict the value of a board 
        positions and their prior distribution over possible next moves.
        """
        # First encode the game states as a tensor which can be passed to the
        # network. Then get the network to make the prediction and decode the
        # network's policy output into a dictionary of move-prior pairs
        encoded_states = [self.encoder.encode(s) for s in game_states]
        input_tensors = [s.reshape(1, 7, 7, 6) for s in encoded_states]
        input_tensor = np.concatenate(input_tensors)
        input_tensor = input_tensor.astype(np.float32)
        
        input_det = self.intrp.get_input_details()[0]
        self.intrp.resize_tensor_input(input_det['index'], input_tensor.shape)
        self.intrp.allocate_tensors()
        
        self.intrp.set_tensor(self.inp_ind, input_tensor)
        self.intrp.invoke()
        priors = self.intrp.get_tensor(self.pol_ind)
        values = self.intrp.get_tensor(self.val_ind)
        
        move_priors = [self.encoder.decode_policy(ps, state.legal_moves()[0]) 
                       for ps, state in zip(priors, game_states)]
        
        predictions = [(priors, value[0]) 
                       for priors, value in zip(move_priors, values)]
        return predictions
    
    def save_network(self, prefix="model_data/"):
        self.model.save(prefix + 'zero_model.h5')
        load_command = "from .networks.zero_network import ZeroNet;"
        load_command += "self.network = ZeroNet()"
        return load_command
        
    def load_network(self, prefix="model_data/"):
        self.model = load_model(prefix + 'zero_model.h5')
        self.compile_lite_model()
        
    def train(self, training_data, batch_size, epochs=1):
        X, Y, rewards = training_data
        loss = self.model.fit(
            X, [Y, rewards],
            batch_size=batch_size, 
            epochs=epochs,
        )
        self.compile_lite_model()
        return loss