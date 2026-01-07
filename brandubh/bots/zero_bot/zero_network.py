#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:16:53 2020

@author: john

This submodule defines classes and functions for building and managing neural 
network models for the ZeroBot class. The neural networks use an architecture
based on the one used by AlphaGo zero.
"""

import tensorflow as tf

import json
import numpy as np
from pathlib import Path

from .encoder import SixPlaneEncoder, ThreePlaneEncoder
from .data_manager import ZeroDataManager, DualDataManager

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.layers import LeakyReLU, add, Softmax, Reshape
from keras.models import load_model
from keras.regularizers import l2
from keras.optimizers import Adam

ENCODERS = {
    'SixPlaneEncoder' : SixPlaneEncoder,
    'ThreePlaneEncoder' : ThreePlaneEncoder
}

class ZeroNet():
    """A class for managing a ZeroBot nerual network.
    """
    
    def __init__(self, network_params = {}, model = None):
        self.model_params = network_params
        self.encoder = ENCODERS[network_params['encoder']]()
        
        if model is not None:
            self.model = model 
        else:
            network_params['input_channels'] = self.encoder.num_planes
            self.model = build_model(**network_params)
        
        self.compile_lite_model()
        
    
    def get_encoder(self):
        return self.encoder
    
    def get_data_manager(self, max_buffer_size, epoch_size):
        return ZeroDataManager(max_buffer_size, epoch_size)
    
    
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
        """Compile the neural network using the Adam optimizer and the given
        output weigths and learning rate (lr).
        """
        self.model.compile(
            optimizer = Adam(learning_rate=lr,),
            loss = ['categorical_crossentropy', 'mse'],
            loss_weights = [policy_weight, value_weight]
        )
    
    
    def predict(self, game_states):
        """Use the neural network to predict the value of the given board 
        positions and their prior distribution over possible next moves.
        """
        if not game_states:
            return []
        
        # Encode the game states as a tensor to be passed to the network.
        encoded_states = [self.encoder.encode(s) for s in game_states]
        input_tensor = np.array(encoded_states, ndmin=4)
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


    def train(self, training_data, batch_size, epochs=1):
        """Train the neural network model on the given data for the given
        number of epochs and using the given batch size.
        """
        X, Y, rewards = training_data
        loss = self.model.fit(
            X, [Y, rewards],
            batch_size=batch_size, 
            epochs=epochs,
        )
        self.compile_lite_model()
        return loss
    
    
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
    
    
class DualNet:
    
    def __init__(self, network_params = {}, models = None):
        if models is None:
            self.white_model = ZeroNet(network_params)
            self.black_model = ZeroNet(network_params)
        else:
            self.white_model, self.black_model = models
        
    
    def get_encoder(self):
        return self.white_model.encoder
    
    def get_data_manager(self, max_buffer_size, epoch_size):
        return DualDataManager(max_buffer_size, epoch_size)
    
    
    def compile_lite_model(self):
        """Creates a lite version of the model for rapid inference.
        """
        self.white_model.compile_lite_model()
        self.black_model.compile_lite_model()
        
    
    def predict(self, game_states):
        """Use the neural network to predict the value of the given board 
        positions and their prior distribution over possible next moves.
        """
        white_states, black_states = self.split_game_states(game_states)
        white_pred = self.white_model.predict(white_states)
        black_pred = self.black_model.predict(black_states)
        return self.merge_predictions(white_pred, black_pred, game_states)
    
    
    def split_game_states(self, game_states):
        white_states = [s for s in game_states if s.player == 1]
        black_states = [s for s in game_states if s.player ==-1]
        return white_states, black_states
    
    def merge_predictions(self, white_pred, black_pred, game_states):
        """Correctly merge the white and black predictions into lists with the
        same order as the given list of gamestates.
        """
        return [
            white_pred.pop(0) if state.player == 1 else black_pred.pop(0)
            for state in game_states
        ]


    def train(self, training_data, batch_size, epochs=1):
        """Train the neural network model on the given data for the given
        number of epochs and using the given batch size.
        """
        white_data, black_data = training_data
        white_loss = self.white_model.train(white_data, batch_size, epochs)
        black_loss = self.white_model.train(black_data, batch_size, epochs)
        self.compile_lite_model()
        return white_loss, black_loss
        
    
    def compile_network(self, policy_weight, value_weight, lr = 0.0001):
        """Compile the neural network using the Adam optimizer and the given
        output weigths and learning rate (lr).
        """
        self.white_model.compile_network(policy_weight, value_weight, lr)
        self.black_model.compile_network(policy_weight, value_weight, lr)
    
    
    def save_network(self, model_dir=Path("model_data/")):
        white_model_dir = model_dir / 'white_model/'
        black_model_dir = model_dir / 'black_model/'
        white_model_dir.mkdir(exist_ok=True)
        black_model_dir.mkdir(exist_ok=True)
        self.white_model.save_network(white_model_dir)
        self.black_model.save_network(black_model_dir)
    
    
    @classmethod
    def load_network(cls, model_dir=Path("model/")):
        white_model = ZeroNet.load_network(model_dir / 'white_model/')
        black_model = ZeroNet.load_network(model_dir / 'black_model/')
        return DualNet(models = (white_model, black_model))
    
    
#=============================================================================#
#                Functions for assembling the neural network
#=============================================================================#

KERNEL_SIZE = (3, 3)

def build_model(
        alpha            = 0.0001,
        input_channels   = 6,
        backbone_depth   = 7,
        backbone_filters = 35,
        value_filters    = 35,
        value_size_a     = 35, 
        value_size_b     = 21,
        policy_filters   = 35,
        **kwargs
    ):
    """Builds the network model for a Brandubh network class. The arhcitecture
    of the model is based on the Alphago zero arhcitecture.
    
    Architecture outline:
        The input layer consists of several 7x7 arrays or planes 
        (see encoder class). This input is processed by a backbone consisting
        of a selected number of residual blocks, each with a selected number
        of filters. (See residual_layer)
        
        The output of the backbone is passed to both a policy head and a value
        head to predict move priors and expected value of the input state
        respectively.
        
        The value head consists of a single convolution layer with a selected 
        filters followed by two dense layers and a final output layer with 
        a single neuron. The sizes of the two hidden layers are determined
        bye the `value_size_a` and `value_size_b` arguments.
        
        The policy head has two consecutive convolutional layers. The 
        `policy_filters` argument determines the number of filters in the first
        convolutional layer and the second has 24 filters. The output of the 
        policy head is a 7x7x24 tensor (see Encoder class) and uses a softmax 
        activation.
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
    input_shape     = (7, 7, input_channels)
    board_input     = Input(shape=input_shape, name='board_input')
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
    """Creates a residual block used in the model's backbone. The bloack
    consists of two convolutional layers using 3x3 kernels, leaky ReLU 
    activations and a skip connection.
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