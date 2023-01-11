#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 12:30:50 2021

@author: john

This file defines a bot that greedily selects random moves to play in brandubh.
"""
import random
from ..brandubh import Act

class GreedyRandomBot:
    """
    This bot looks at all legal moves it can make and randomly chooses a move
    that results in a win. If no so move exists a random move is selected.
    """
    is_trainable = False
    
    def select_move(self, game_state):
        """Choose a random valid move."""
        
        # Get a list of possible moves
        candidates = game_state.legal_moves()
        
        # If there's no candidate moves then pass the turn
        if not candidates:
            return Act.pass_turn()
        
        # Get all moves that result in a win. If there are any, randomly select
        # one of those moves.
        winning_moves = self.get_winning_moves(candidates, game_state)
        if len(winning_moves) > 0:
            return Act.play(random.choice(winning_moves))
        
        # return a random move from the list of candidates
        return Act.play(random.choice(candidates))
    
    def get_winning_moves(self, candidate_moves, game_state):
        """
        Returns a list of moves from candidate_moves that result in a win.
        """
        winning_moves = []
        for move in candidate_moves:
            next_state = game_state.copy()
            next_state.take_turn_with_no_checks(Act.play(move))
            if not next_state.winner == 0:
                winning_moves.append(move)
                
        return winning_moves