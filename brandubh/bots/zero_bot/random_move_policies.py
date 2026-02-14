#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 19:40:56 2026

@author: john

This submodule defines RandomMovePolicy classes that can be used to decide when
a random move should be played during a sel play game.
"""

import numpy as np


class RandomMovePolicy:
    """Base interface for random move policies."""
    def make_random_move(self, game_state):
        raise NotImplementedError
        
    def update(self, training_logs):
        pass
    
    def get_state_dict(self):
        raise NotImplementedError



class ConstantEpsilon(RandomMovePolicy):
    """A random move policy object to implement a constant random move rate."""
    def __init__(self, eps=0.07):
        self.eps = eps
        
    def make_random_move(self, game_state):
        return np.random.rand() < self.eps
    
    def get_state_dict(self):
        return {
            'random_move_policy' : 'ConstantEpsilon',
            'eps': self.eps
        }



class BalancedEpsilon(RandomMovePolicy):
    """A random move policy that dynamically adjusts the random move rate based 
    on how balanced the self play experience is.
    """
    def __init__(self, window_size=21, initial_eps=0.0, gain=0.01):
        self.window_size = window_size
        self.curr_eps = initial_eps
        self.gain = gain
        
        self.eps_history = [initial_eps]
        self.curr_window_ind = 0
        self.balance_window = {
            'white_window' : [0] * window_size,
            'black_window' : [0] * window_size,
            'draws_window' : [0] * window_size
        }
    
    
    def update(self, training_logs):
        logs = training_logs['balance_history']
        ind = self.curr_window_ind
        self.balance_window['white_window'][ind] = logs['white_win_moves'][-1]
        self.balance_window['black_window'][ind] = logs['black_win_moves'][-1]
        self.balance_window['draws_window'][ind] = logs['draw_moves'][-1]
        self.curr_window_ind = (self.curr_window_ind + 1) % self.window_size
        self.update_eps()
    
    
    def update_eps(self):
        total_white = sum(self.balance_window['white_window'])
        total_black = sum(self.balance_window['black_window'])
        total_draws = sum(self.balance_window['draws_window'])
        
        balance_score = (total_white - total_black)
        balance_score /= (total_white + total_black + total_draws)
        self.curr_eps += self.gain * balance_score
        self.eps_history.append(self.curr_eps)
    
    
    def make_random_move(self, game_state):
        # Should only make a random move if player and eps have the same sign.
        if (self.curr_eps * game_state.player) > 0:
            return np.random.rand() < abs(self.curr_eps)
        return False
    
    
    def get_state_dict(self):
        return {
            'random_move_policy' : 'BalancedEpsilon',
            'gain': self.gain,
            'window_size': self.window_size,
            'eps_history': self.eps_history
        }