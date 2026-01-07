#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 20:24:04 2026

@author: john

This submodule define functions for assembling the neural network for a 
brandubh ZeroBot. The neural networks use an architecture based on the one used 
by AlphaGo zero.
"""

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.layers import LeakyReLU, add, Softmax, Reshape
from keras.regularizers import l2

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