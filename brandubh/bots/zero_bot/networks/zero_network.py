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

import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import copy

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.layers import LeakyReLU, add, Softmax, Reshape
from keras.models import load_model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, LearningRateScheduler

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
    
    def __init__(self):
        self.model = ZeroNet.build_model()
        self.compile_lite_model()
        self.encoder = SixPlaneEncoder()
        self.loss_history = []
        self.training_rounds = 0
        self.batch_size = 0
        
        self.stop_criteria = EarlyStopping(monitor="loss",
                                           patience=7,
                                           mode="min",
                                           restore_best_weights=True)
        
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
        
        x = Conv2D(filters, kernel_size, use_bias = True,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x
    
    @classmethod
    def residual_layer(cls, input_block, filters, kernel_size):
        biases = True
        
        x = ZeroNet.conv_layer(input_block, filters, kernel_size)
        x = Conv2D(filters, kernel_size, use_bias = biases,
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
        biases = True
        x = Conv2D(filters = 21, kernel_size = (1, 1), use_bias = True,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        # x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(21, use_bias = biases, 
                  activation = 'linear',
                  bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        x = LeakyReLU()(x)
        x = Dense(14, use_bias = biases, 
                  activation = 'linear',
                  bias_regularizer = l2(cls.alpha),
                  kernel_regularizer = l2(cls.alpha))(x)
        x = LeakyReLU()(x)
        x = Dense(1, use_bias = biases, activation = 'tanh', 
                  name = 'value_head')(x)
        return x
    
    @classmethod
    def policy_head(cls, x):
        biases = True
        x = Conv2D(filters = 70, kernel_size = (1, 1), use_bias = biases,
                   padding = 'same', 
                   activation = 'linear',
                   bias_regularizer = l2(cls.alpha),
                   kernel_regularizer = l2(cls.alpha))(x)
        x = LeakyReLU()(x)
        
        x = Conv2D(filters = 24, kernel_size = (1, 1), use_bias = biases,
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
        
        processed_board = ZeroNet.conv_layer(board_input, 28, (3, 3))
        for i in range(7):
            processed_board = ZeroNet.residual_layer(processed_board, 28, (3, 3))
            
        policy_output = ZeroNet.policy_head(processed_board)
        value_output = ZeroNet.value_head(processed_board)
        
        model = Model(inputs=board_input,
                      outputs=[policy_output, value_output])
        return model
    
    def predict(self, game_states):
        """
        This method uses the neural network to predict the value of a board 
        positions and their prior distribution over possible next moves.
        """
        # First encode the game states as a tensor which can be passed to the
        # network. Then get the network to make the prediction and decode the
        # network's policy output into a dictionary of move-prior pairs
        encoded_states = [self.encoder.encode(s, True) for s in game_states]
        input_tensors = [s[0].reshape(1, 7, 7, 6) for s in encoded_states]
        input_tensor = np.concatenate(input_tensors)
        input_tensor = input_tensor.astype(np.float32)
        
        if self.batch_size != input_tensor.shape[0]:
            self.batch_size = input_tensor.shape[0]
            input_det = self.intrp.get_input_details()[0]
            self.intrp.resize_tensor_input(input_det['index'], input_tensor.shape)
            self.intrp.allocate_tensors()
        
        self.intrp.set_tensor(self.inp_ind, input_tensor)
        self.intrp.invoke()
        priors = self.intrp.get_tensor(self.pol_ind)
        values = self.intrp.get_tensor(self.val_ind)
        
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
    
    def print_prediction(self, input_tensor):
        """
        This method uses the neural network to predict the value of a board 
        position and the prior distribution over possible next moves.
        """
        
        input_tensor = input_tensor.reshape(1, 7, 7, 6)
        input_tensor = input_tensor.astype(np.float32)
        
        self.intrp.set_tensor(self.inp_ind, input_tensor)
        self.intrp.invoke()
        priors = self.intrp.get_tensor(self.pol_ind)
        value = self.intrp.get_tensor(self.val_ind)
        
        header = "##########- input tensor ~##########"
        print_tensor(input_tensor, header)
        
        header = "##########- output tensor - value-head = {0} ~##########".format(value)
        print_tensor(priors, header)
    
    def save_network(self, prefix="model_data/"):
        self.model.save(prefix + 'zero_model.h5')
        load_command = "from bots.zero_bot.zero_network import ZeroNet; "
        load_command += "self.network = ZeroNet()"
        return load_command
        
    def load_network(self, prefix="model_data/"):
        self.model = load_model(prefix + 'zero_model.h5')
        self.compile_lite_model()
        
    def train(self, training_data, batch_size, epochs=1):
        X, Y, rewards = training_data
        lr_schedule = self.get_lr_schedule()
        loss = self.model.fit(X, [Y, rewards],
                              batch_size=batch_size, 
                              epochs=epochs,
                              callbacks=[self.stop_criteria, lr_schedule])
        self.compile_lite_model()
        self.save_losses(loss)
        
    def get_lr_schedule(self):
        n = len(self.loss_history)
        # if n < 70:
        #     schedule = lambda epoch, lr : (1/7)**(7)
        # else:
        schedule = lambda epoch, lr : (1/7)**(4 + (n + epoch)//350)
        return LearningRateScheduler(schedule)
    
    
    def save_losses(self, loss_history):
        """
        Method to save the evaulations of the loss function of the neural
        network on training data.
        """
        ls = list(loss_history.history.values())
        losses = [[l[i] for l in ls] for i in range(len(ls[0]))]
        for l in losses:
            l.append(self.training_rounds)
            self.loss_history.append(l)
        self.training_rounds += 1
        
    def num_epochs(self):
        return len(self.loss_history)
    
    def create_training_data(self, experience):
        """
        A method to convert game data in an experience list to training data
        for training the ZeroBot neural network. The training data is also
        expanded 8 fold using symetries of the game. 
        """
        
        # The network input and labels forming the training set will be stored 
        # in the following lists.
        X, Y, rewards = [], [], []
        
        # For each episode in the experience append the relevant tensors to 
        # the X, Y and reward lsits.
        for episode in experience:
            
            Xi =  np.array(episode['boards']) 
            num_moves = Xi.shape[0]
            X.append(Xi)
            
            visit_counts = episode['prior_targets']
            policy_targets = self.encoder.encode_priors(visit_counts)
            
            # The reward for moves decays exponentially with the number of
            # moves between it and the winning move. Rewards for moves made by 
            # the winning side are positive and negative for the losing side.
            episode_rewards = episode['winner'] * np.array(episode['players'])
            # episode_rewards = (np.exp(-1*(num_moves-np.arange(num_moves)-1)/28
            #                           )) * episode_rewards
            
            rewards.append( episode_rewards )
            
            Y.append( policy_targets )
          
        # Convert the X, Y lists into numpy arrays
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        
        # Use the bot's game encoder to expand the training data 8 fold.
        X, Y = self.encoder.expand_data(X, Y)
        rewards = np.concatenate(8*rewards)
            
        return X, Y, rewards
    
    
class SixPlaneEncoder():
    """
    This class is used to encode a brandubh game state as a tensor which can
    be fed into the neural network created by the ZeroNet class. It also has
    methods for decoding the output tensor from the policy head of the network
    into a dictionary of move-prior pairs, encoding a prior distribution as
    a tensor with the same shape as the policy head output (used for creating
    training data), and for expanding a training data set using symmetries of
    the game board.
    """
        
    def encode(self, game_state, return_pieces=False):
        """
        A game state is encoded as six 7x7 planes (or a tensor with shape 
        (7,7,6)) as described below.
        
        The first 7x7 plane is an array of 0's and 1's encoding the positions 
        of all pieces owned by the current player, i.e. the array has a 1 in 
        entries corresponding to squares of the board occupied by one of the 
        current players pieces and a 0 otherwise.
        
        The second 7x7 plane is an array encoding the position of king piece 
        owned by the current player. If the player doesn't own a king then 
        this plane will be all zeros, otherwise it will have a 1 in the 
        relevant entry.
        
        The next two planes are the same as the first two but of the opposite 
        player.
        
        The fifth plane is a 7x7 array of 1's if the current player is playing 
        as white and is all 0's otherwise.
        
        The sixth plane is a 7x7 array of 1's if the current player is playing 
        as black and is all 0's otherwise.
        """
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
            board_tensor[r,c,1 - player] = 1
            if piece == 2:
                board_tensor[r,c,2 - player] = 1
            
        for (r, c) in game_state.game_set.black_pieces:
            board_tensor[r,c,1 + player] = 1
        
        if return_pieces:
            return board_tensor, pieces
        return board_tensor
    
    def decode_policy(self, model_output, pieces):
        """
        The policy head of the ZeroNet outputs a tensor with shape (7,7,24)
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
    
def print_tensor(tensor, header, filename='prediction_output.txt'):
    with open(filename, 'a') as f:
        f.write(header + "\n\n\n")
        
        shape = tensor.shape
        if (not len(shape) == 4) and (not shape[0] == 1):
            f.write("Tensor has the wrong shape")
            
        else:
            for plane in range(shape[3]):
                f.write("Plane {0}\n".format(plane))
                for i in range(shape[1]):
                    row = ""
                    for j in range(shape[2]):
                        row += " {0:.0E} ".format(tensor[0, i, j, plane])
                    f.write(row + "\n")
                f.write("\n")
            f.write("The sum of the elements is {0}".format(tensor.sum()))
            f.write("\n\n")