#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:52:41 2019

@author: john

This file defines a bot that plays brandubh by playing random moves each turn.
"""

import random
from brandubh import Act

class RandomBot:
    """
    This is a bot that will play a random move in brandubh when asked to
    make a move
    """
    
    def select_move(self, game_state):
        """Choose a random valid move."""
        
        # Get a list of possible moves
        candidates = game_state.legal_moves()
        
        # If there's no candidate moves then pass the turn
        if not candidates:
            return Act.pass_turn()
        
        # return a random move from the list of candidates
        return Act.play(random.choice(candidates))