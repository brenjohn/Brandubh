#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 16:09:30 2021

@author: john

This file contains two classes defining a neural network which can be used
by a ZeroBot. The classes implement the interface a ZeroBot needs to interact 
with the neural network in order to predict the value of a board position and 
a distribution over moves that can be made from a board position.

The DualNet class has a keras neural network for predicting the value of a
board-position/game-state and the distribution of visits the ZeroBot
select move algorithm will make to branches of the decision tree stemming 
from the board-position.

The ThreePlaneEncoder class is used to convert a game-state to an input tensor
for the neural network and to convert the output of the network to a 
dictionary of move-value pairs. It also has methods for expanding training 
data for the network.
"""
# Disable tensorflow logging messages:
import logging
import os
logging.disable(logging.INFO)
logging.disable(logging.WARNING)
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import copy

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.layers import LeakyReLU, add
from keras.models import load_model
from keras.optimizers import Adam

import tensorflow as tf


class DualNet():
    """
    This class has methods for building, saving and loading nerual networks
    to make predictions required by a Zerobot.
    
    The network architecture is as follows:
        The input layer consists of 3 7x7 arrays of neurons (see encoder class)
        
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
    # Regularisation constant
    alpha = 0.005
    
    def __init__(self):
        self.black_model = DualNet.build_model()
        self.white_model = DualNet.build_model()
        
        self.compile_lite_models()
        
        self.encoder = ThreePlaneEncoder()
        self.loss_history_b = []
        self.loss_history_w = []
        
    def compile_lite_models(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.black_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        black_model = converter.convert()
        self.black_intrp = tf.lite.Interpreter(model_content=black_model)
        self.black_intrp.allocate_tensors()
        input_det = self.black_intrp.get_input_details()[0]
        policy_det, value_det = self.black_intrp.get_output_details()
        self.black_inp_ind = input_det["index"]
        self.black_val_ind = value_det["index"]
        self.black_pol_ind = policy_det["index"]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.white_model)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        white_model = converter.convert()
        self.white_intrp = tf.lite.Interpreter(model_content=white_model)
        self.white_intrp.allocate_tensors()
        input_det = self.white_intrp.get_input_details()[0]
        policy_det, value_det = self.white_intrp.get_output_details()
        self.white_inp_ind = input_det["index"]
        self.white_val_ind = value_det["index"]
        self.white_pol_ind = policy_det["index"]
        
    @classmethod
    def conv_layer(cls, x, filters, kernel_size):
        
        x = Conv2D(filters, kernel_size, use_bias = True,
                   padding = 'same', activation = 'linear')(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x
    
    @classmethod
    def residual_layer(cls, input_block, filters, kernel_size):
        
        x = DualNet.conv_layer(input_block, filters, kernel_size)
        x = Conv2D(filters, kernel_size, use_bias = True,
                   padding = 'same', activation = 'linear')(x)
        # x = BatchNormalization(axis=1)(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x
    
    @classmethod
    def value_head(cls, x):
        
        x = Conv2D(filters = 14, kernel_size = (3, 3), use_bias = True,
                   padding = 'same', activation = 'linear')(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters = 14, kernel_size = (3, 3), use_bias = True,
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
        """
        This method builds the network model when a DualNet is initialised.
        """
        board_input = Input(shape=(7,7,3), name='board_input')
        
        processed_board = DualNet.conv_layer(board_input, 14, (3, 3))
        for i in range(7):
            processed_board = DualNet.residual_layer(processed_board, 14, (3, 3))
            
        value_output = DualNet.value_head(processed_board)
        policy_output = DualNet.policy_head(processed_board)
        
        model = Model(inputs=board_input, 
                      outputs=[policy_output, value_output])
        return model
    
    def compile_network(self, policy_weight, value_weight):
        self.white_model.compile(optimizer = Adam(),
                                 loss = ['categorical_crossentropy', 'mse'],
                                 loss_weights = [policy_weight, value_weight])
        self.black_model.compile(optimizer = Adam(),
                                 loss = ['categorical_crossentropy', 'mse'],
                                 loss_weights = [policy_weight, value_weight])
    
    def predict(self, game_states):
        """
        This method uses the neural networks to predict the value of a board 
        position and the prior distribution over possible next moves.
        """
        # First encode the game state as a tensor which can be passed to the
        # network. Then get the network to make the prediction and decode the
        # network's policy output into a dictionary of move-prior pairs
        white_states = [self.encoder.encode(s, True) for s in game_states 
                        if s.player == 1]
        white_tensors = [s[0].reshape(1, 7, 7, 6) for s in white_states]
        white_tensor = np.concatenate(white_tensors)
        white_tensor = white_tensor.astype(np.float32)
        
        black_states = [self.encoder.encode(s, True) for s in game_states 
                        if s.player ==-1]
        black_tensors = [s[0].reshape(1, 7, 7, 6) for s in black_states]
        black_tensor = np.concatenate(black_tensors)
        black_tensor = black_tensor.astype(np.float32)
        
        if black_tensor:
            if self.batch_size != black_tensor.shape[0]:
                self.batch_size = black_tensor.shape[0]
                input_det = self.intrp.get_input_details()[0]
                self.intrp.resize_tensor_input(input_det['index'], black_tensor.shape)
                self.intrp.allocate_tensors()
            
            self.black_intrp.set_tensor(self.black_inp_ind, black_tensor)
            self.black_intrp.invoke()
            black_priors = self.black_intrp.get_tensor(self.black_pol_ind)
            black_values = self.black_intrp.get_tensor(self.black_val_ind)
            
        if white_tensor:
            if self.batch_size != white_tensor.shape[0]:
                self.batch_size = white_tensor.shape[0]
                input_det = self.intrp.get_input_details()[0]
                self.intrp.resize_tensor_input(input_det['index'], white_tensor.shape)
                self.intrp.allocate_tensors()
            
            self.white_intrp.set_tensor(self.white_inp_ind, white_tensor)
            self.white_intrp.invoke()
            white_priors = self.white_intrp.get_tensor(self.white_pol_ind)
            white_values = self.white_intrp.get_tensor(self.white_val_ind)
        
        
        priors = None # merge white and black priors correctly
        values = None # merge white and black values correctly
        encoded_states = None # merge white and black encoded states correctly
        
        states_pieces = [s[1] for s in encoded_states]
        move_priors = [self.encoder.decode_policy(p, pieces) 
                       for p, pieces in zip(priors, states_pieces)]
        
        # Restrict and noramlise prior distributions over possible moves.
        for priors in move_priors:
            N = sum(priors.values())
            for (move, prior) in priors.items():
                priors[move] /= N
        
        predictions = [(priors, value[0]) 
                       for priors, value in zip(move_priors, values)]
        
        return predictions
    
    def save_network(self, prefix="model_data/"):
        self.black_model.save(prefix + 'black_model.h5')
        self.white_model.save(prefix + 'white_model.h5')
        load_command = "from bots.zero_bot.dual_network import DualNet; "
        load_command += "self.network = DualNet()"
        return load_command
        
    def load_network(self, prefix="model_data/"):
        self.black_model = load_model(prefix + 'black_model.h5')
        self.white_model = load_model(prefix + 'black_model.h5')
        self.compile_lite_models()
        
    def num_epochs(self):
        return len(self.loss_history_w)
        
    def create_training_data(self, experience):
        """
        A method to convert game data in an experience list to training data
        for training the ZeroBot neural network. The training data is also
        expanded 8 fold using symetries of the game. 
        """
        
        # The network input and labels forming the training set will be stored 
        # in the following lists.
        X_b, Y_b, rewards_b = [], [], []
        X_w, Y_w, rewards_w = [], [], []
        
        # For each episode in the experience append the relevant tensors to 
        # the X, Y and reward lsits.
        for episode in experience:
            
            Xi =  np.array(episode['boards']) 
            # num_moves = Xi.shape[0]
            
            visit_counts = episode['prior_targets']
            policy_targets = self.encoder.encode_priors(visit_counts)
            
            # The reward for moves decays exponentially with the number of
            # moves between it and the winning move. Rewards for moves made by 
            # the winning side are positive and negative for the losing side.
            episode_rewards = episode['winner'] * np.array(episode['players'])
            # episode_rewards *= 1*(np.arange(num_moves)-(num_moves-41) > 0)
            
            for n, player in enumerate(episode['players']):
                if player == -1:
                    X_b.append(Xi[n])
                    Y_b.append(policy_targets[n])
                    rewards_b.append(episode_rewards[n])
                    
                else:
                    X_w.append(Xi[n])
                    Y_w.append(policy_targets[n])
                    rewards_w.append(episode_rewards[n])
          
        # Convert the X, Y lists into numpy arrays
        X_b = np.stack(X_b)
        X_w = np.stack(X_w)
        Y_b = np.stack(Y_b)
        Y_w = np.stack(Y_w)
        
        # Use the bot's game encoder to expand the training data 8 fold.
        X_b, Y_b = self.encoder.expand_data(X_b, Y_b)
        rewards_b = np.stack(8*rewards_b)
        
        X_w, Y_w = self.encoder.expand_data(X_w, Y_w)
        rewards_w = np.stack(8*rewards_w)
            
        return X_b, X_w, Y_b, Y_w, rewards_b, rewards_w
    
    def train(self, training_data, batch_size, epochs):
        X_b, X_w, Y_b, Y_w, rewards_b, rewards_w = training_data
        loss_b = self.black_model.fit(X_b, [Y_b, rewards_b], 
                             batch_size=batch_size, epochs=epochs)
        loss_w = self.white_model.fit(X_w, [Y_w, rewards_w], 
                             batch_size=batch_size, epochs=epochs)
        self.compile_lite_model()
        return loss_b, loss_w
    
    def save_losses(self, loss_history):
        """
        Method to save the evaulations of the loss function of the neural
        network on training data.
        """
        losses_history_b, losses_history_w = loss_history
        losses_b = [loss[0] for loss in losses_history_b.history.values()]
        losses_w = [loss[0] for loss in losses_history_w.history.values()]
        self.loss_history_b.append(losses_b)
        self.loss_history_w.append(losses_w)
    
    
class ThreePlaneEncoder():
    """
    This class is used to encode a brandubh game state as a tensor which can
    be fed into the neural network created by the DualNet class. It also has
    methods for decoding the output tensor from the policy head of the network
    into a dictionary of move-prior pairs, encoding a prior distribution as
    a tensor with the same shape as the policy head output (used for creating
    training data), and for expanding a training data set using symmetries of
    the game board.
    """
    def __init__(self):
        self.convert_to_tensor_element = {1: 'board_tensor[r,c,0] = 1',
                                          2: 'board_tensor[r,c,1] = 1',
                                         -1: 'board_tensor[r,c,2] = 1'}
        
    def encode(self, game_state, return_pieces=False):
        """
        A game state is encoded as three 7x7 planes (or a tensor with shape 
        (7,7,3)) as described below.
        
        The first 7x7 plane is an array of 0's and 1's encoding the positions 
        of all soldier pieces owned by the white player, i.e. the array has a 
        1 in entries corresponding to squares of the board occupied by a white
        soldier.
        
        The second 7x7 plane is an array encoding the position of king piece.
        
        The third plane encodes the position of all the black pieces.
        """
        board_tensor = np.zeros((7,7,3))
        player = game_state.player
        
        if player == 1:
            pieces = game_state.game_set.white_pieces
        else:
            pieces = game_state.game_set.black_pieces
        
        for (r, c) in game_state.game_set.white_pieces:
            piece = game_state.game_set.board[(r, c)]
            board_tensor[r,c,piece-1] = 1
            # exec(self.convert_to_tensor_element[piece])
            
        for (r, c) in game_state.game_set.black_pieces:
            board_tensor[r,c,2] = 1
            # exec(self.convert_to_tensor_element[-1])
        
        if return_pieces:
            return board_tensor, pieces
        return board_tensor
    
    def decode_policy(self, model_output, pieces):
        """
        The policy head of the DualNet outputs a tensor with shape (7,7,24)
        containing a probaility distribution over possible moves to make.
        
        The first two indices of the tensor correspond to a square on the 
        board and the third index indicates a possible move that a piece at 
        that square could possibly make.
        
        Values of the third index ranging from 0 to 5 correspond to decreasing
        the y coordinate of the piece by 6 to 1 places respectively.
        
        Values of the third index ranging from 6 to 11 correspond to increasing
        the y coordinate of the piece by 1 to 6 places respectively.
        
        Values of the third index ranging from 12 to 23 similarly correspond to
        either decreasing or increasing the x coordinate of the piece.
        
        So the prior for moving the piece at square (3, 2) 2 places to (3, 4)
        is output_tensor[3, 2, 7].
        """
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
        Y[:,:,:,12:] = Y[:,:,:,24:11:-1]
        return Y