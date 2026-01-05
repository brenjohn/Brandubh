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

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Normal imports:
import json
import numpy as np
from pathlib import Path

from .encoder import SixPlaneEncoder

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.layers import LeakyReLU, add, Softmax, Reshape
from keras.models import load_model
from keras.regularizers import l2
from keras.optimizers import Adam


class ZeroNet():
    """A class for managing a zerobot nerual network.
    
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
    
    def __init__(self, network_params = {}, model = None):
        self.model_params = network_params
        model_provided = model is not None
        self.model = model if model_provided else build_model(**network_params)
        
        self.encoder = SixPlaneEncoder()
        self.compile_lite_model()
        
    def get_encoder(self):
        return self.encoder
    
    
    def compile_lite_model(self):
        """Creates a lite version of the model for rapid inference.
        """
        # Create lite model.
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        lite_model = converter.convert()
        
        # Get model interpreter and allocate associated tensors.
        self.intrp = tf.lite.Interpreter(model_content=lite_model)
        self.intrp.allocate_tensors()
        
        # Get indices of input, value and policy tensors.
        input_det = self.intrp.get_input_details()[0]
        policy_det, value_det = self.intrp.get_output_details()
        self.input_size  = input_det["shape"][0]
        self.input_ind   = input_det["index"]
        self.value_ind   = value_det["index"]
        self.policy_ind  = policy_det["index"]
    
    
    def compile_network(self, policy_weight, value_weight, lr = 0.0001):
        self.model.compile(
            optimizer = Adam(learning_rate=lr,),
            loss = ['categorical_crossentropy', 'mse'],
            loss_weights = [policy_weight, value_weight]
        )
    
    
    def predict(self, game_states):
        """Use the neural network to predict the value of the given board 
        positions and their prior distribution over possible next moves.
        """
        # Encode the game states as a tensor to be passed to the network.
        encoded_states = [self.encoder.encode(s) for s in game_states]
        input_tensor = np.array(encoded_states)
        input_tensor = input_tensor.astype(np.float32)
        
        # Resize the input tensors if needed.
        if input_tensor.shape[0] != self.input_size:
            self.intrp.resize_tensor_input(self.input_ind, input_tensor.shape)
            self.intrp.allocate_tensors()
            self.input_size = input_tensor.shape[0]
        
        # Use the neural network to make predictions for the inputs.
        self.intrp.set_tensor(self.input_ind, input_tensor)
        self.intrp.invoke()
        priors = self.intrp.get_tensor(self.policy_ind)
        values = self.intrp.get_tensor(self.value_ind)
        
        # Decode and return predictions.
        move_priors = [
            self.encoder.decode_policy(ps, state.legal_moves()[0]) 
            for ps, state in zip(priors, game_states)
        ]
        return [
            (priors, value[0]) for priors, value in zip(move_priors, values)
        ]
    
    
    def save_network(self, model_dir=Path("model_data/")):
        with open(model_dir / 'network_params.json', 'w') as file:
            json.dump(self.model_params, file, indent=4)
        self.model.save(model_dir / 'zero_model.h5')
    
    
    @classmethod
    def load_network(cls, model_dir=Path("model/")):
        with open(model_dir / 'params.json', 'w') as file:
            model_params = json.load(file)
        model = load_model(model_dir / 'zero_model.h5')
        network = ZeroNet(model_params, model)
        network.compile_lite_model()
        return network
    
    
    def train(self, training_data, batch_size, epochs=1):
        X, Y, rewards = training_data
        loss = self.model.fit(
            X, [Y, rewards],
            batch_size=batch_size, 
            epochs=epochs,
        )
        self.compile_lite_model()
        return loss
    
    
#=============================================================================#
#                Functions for assembling the neural network
#=============================================================================#

KERNEL_SIZE = (3, 3)

def build_model(
        alpha            = 0.0001, 
        backbone_depth   = 7,
        backbone_filters = 35,
        value_filters    = 35,
        value_size_a     = 35, 
        value_size_b     = 21,
        policy_filters   = 35
    ):
    """Builds the network model for a Brandubh network class. The arhcitecture
    of the model is based on the Alphago zero arhcitecture. 
    """
    # Prepare arguments for model components.
    kwargs = {
        'use_bias'           : True,
        'padding'            : 'same', 
        'activation'         : 'linear',
        'bias_regularizer'   : l2(alpha),
        'kernel_regularizer' : l2(alpha)
    }
    backbone_args = (backbone_depth, backbone_filters)
    policy_head_args = (policy_filters,)
    value_head_args = (value_filters, value_size_a, value_size_b)
    
    # Assemble the modle.
    board_input     = Input(shape=(7,7,6), name='board_input')
    backbone_output = backbone(board_input, *backbone_args, kwargs)
    policy_output   = policy_head(backbone_output, *policy_head_args, kwargs)
    value_output    = value_head(backbone_output, *value_head_args, kwargs)
    
    return Model(inputs=board_input, outputs=[policy_output, value_output])


def backbone(x, depth, filters, kwargs):
    """Creates the backbone of the neural network.
    """
    x = Conv2D(filters, KERNEL_SIZE, **kwargs)(x)
    x = LeakyReLU()(x)
    for i in range(depth):
        x = residual_layer(x, filters, kwargs)
    return x


def residual_layer(x, filters, kwargs):
    """Creates a residual block used in the model's backbone.
    """
    y = Conv2D(filters, KERNEL_SIZE, **kwargs)(x)
    y = LeakyReLU()(y)
    y = Conv2D(filters, KERNEL_SIZE, **kwargs)(y)
    x = add([x, y])
    x = LeakyReLU()(x)
    return x


def value_head(x, filters, size_a, size_b, kwargs):
    """Creates hte model's value head.
    """
    dense_kwargs = kwargs.copy()
    dense_kwargs.pop('padding', None)
    final_kwargs = {**dense_kwargs, 'activation' : 'tanh'}
    
    x = Conv2D(filters=filters, kernel_size=(1,1), **kwargs)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    x = Dense(size_a, **dense_kwargs)(x)
    x = LeakyReLU()(x)
    x = Dense(size_b, **dense_kwargs)(x)
    x = LeakyReLU()(x)
    return Dense(1, name = 'value_head', **final_kwargs)(x)


def policy_head(x, filters, kwargs):
    """Creates the model's policy head.
    """
    x = Conv2D(filters, kernel_size=(1,1), **kwargs)(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=24, kernel_size=(1,1), **kwargs)(x)
    x = Reshape(target_shape = (-1,))(x)
    x = Softmax()(x)
    return Reshape(target_shape = (7, 7, 24), name='policy_head')(x)