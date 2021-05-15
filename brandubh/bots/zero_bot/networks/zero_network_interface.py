#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:50:55 2020

@author: john
"""

class net():
    
    def __init__(self):
        self.encoder = Encoder()
        self.model = self.create_model()
        
    def create_model(self):
        """function to create a model for making prediction"""
        pass
    
    def predict(self, model_input):
        """function to pass input to the model and return its output"""
        # Should return a dictionary of move-prior pairs and the value from
        # the network's value head
        pass
    
    def save_network(self):
        # Should save the model and return a string containing a command
        # that can be used for creating the correct net object with correct
        # Encoder object.
        pass
    
    def load_network(self):
        pass
        
        
        
def Encoder():
    
    def __init__(self):
        pass
    
    def encode(self, game_state):
        # Should return a tensor representing the game_state that can be
        # passed to the network model as input
        pass
    
    def encode_priors(self, priors):   
        # Should return a tensor representing the given probability
        # distribution (created from visit counts) over moves which can be
        # used as a training label for the network model
        pass
    
    
    def expand_data(self, X, Y):
        # Should increase the size of the training data X and Y using board
        # symmeteries. Can just return X, Y to avoid implementing this.
        pass
        