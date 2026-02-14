#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:16:53 2020

@author: john

This submodule defines classes for managing neural network models for the 
ZeroBot class.
"""

import tensorflow as tf

import json
import numpy as np
from pathlib import Path
from keras.models import load_model
from keras.optimizers import Adam

from .network import build_model
from .encoder import SixPlaneEncoder, ThreePlaneEncoder
from .data_manager import ZeroDataManager, DualDataManager


ENCODERS = {
    'SixPlaneEncoder' : SixPlaneEncoder,
    'ThreePlaneEncoder' : ThreePlaneEncoder
}



class ZeroNet():
    """A class for managing a ZeroBot nerual network.
    """
    
    def __init__(self, network_params = {}, model = None):
        self.model_params = network_params
        
        if 'encoder' in network_params:
            self.encoder = ENCODERS[network_params['encoder']]()
        else:
            self.encoder = SixPlaneEncoder()
        
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
        return {
            key : [float(val) for val in values] 
            for key, values in loss.history.items()
        }
    
    
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
    """A for managing two ZeroNet objects responsible for processing white
    and black player turns respectively.
    """
    
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
        """Seperate the given game states into two lists holding states where 
        the white and black player is moving respectively. 
        """
        white_states = [s for s in game_states if s.player == 1]
        black_states = [s for s in game_states if s.player ==-1]
        return white_states, black_states
    
    def merge_predictions(self, white_pred, black_pred, game_states):
        """Correctly merge the white and black predictions into lists with the
        same order as the given list of gamestates.
        """
        white_pred, black_pred = iter(white_pred), iter(black_pred)
        return [
            next(white_pred) if state.player == 1 else next(black_pred)
            for state in game_states
        ]


    def train(self, training_data, batch_size, epochs=1):
        """Train the neural network model on the given data for the given
        number of epochs and using the given batch size.
        """
        white_data, black_data = training_data
        white_loss = self.white_model.train(white_data, batch_size, epochs)
        black_loss = self.black_model.train(black_data, batch_size, epochs)
        self.compile_lite_model()
        return {'white_loss' : white_loss, 'black_loss' : black_loss}
        
    
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