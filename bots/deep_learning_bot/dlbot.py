#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:07:46 2020

@author: john
"""

import numpy as np
from brandubh import Act
from four_plane_encoder import FourPlaneEncoder
from keras.models import load_model


class DeepLearningBot:
    
    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            print('loading deep learning bot')
            self.model = self.load_prediction_agent('dl_model_temp.h5')
        self.encoder = FourPlaneEncoder()
        
    def predict(self, game_state):
        input_tensor = self.encoder.encode(game_state)
        input_tensor = input_tensor.reshape(1,7,7,4)
        return self.model.predict(input_tensor)[0], input_tensor
    
    def select_move(self, game_state):
        num_moves = 96
        move_probs, input_tensor = self.predict(game_state)
        
        move_probs = move_probs**3
        move_probs /= np.sum(move_probs)
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1-eps)
        move_probs /= np.sum(move_probs)
        
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates,
                                        num_moves,
                                        replace=False,
                                        p=move_probs)
        
        pieces = np.sum(input_tensor[:,:,:,:2],-1).reshape(7,7)
        for move_idx in ranked_moves:
            move = self.encoder.decode_move_index(pieces, move_idx)
            if not game_state.is_move_illegal(move):
                    return Act.play(move)
        return Act.pass_turn()
    
    # def serialize(self, h5file):
    #     h5file.create_group('encoder')
    #     h5file['encoder'].attrs['name'] = self.encoder.name()
    #     h5file['encoder'].attrs['board_width'] = self.encoder.board_width
    #     h5file['encoder'].attrs['board_height']= self.encoder.board_height
    #     h5file.create_group('model')
    #     kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])
        
    def load_prediction_agent(self, h5file):
        # model = kerasutil.load_model_from_hdf5_group(h5file['model'])
        # encoder_name = h5file['encoder'].attrs['name']
        # if not isinstance(encoder_name, str):
        #     encoder_name = encoder_name.decode('ascii')
        # board_width = h5file['encoder'].attrs['board_width']
        # board_height= h5file['encoder'].attrs['board_height']
        # encoder = encoders.get_encoder_by_name(encoder_name, 
        #                                        (board_width, board_height))
        return load_model(h5file)