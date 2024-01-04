#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 12:30:50 2021

@author: john

This file defines a bot that greedily selects random moves to play in brandubh.
"""
import random
from ..game import Act

class GreedyRandomBot:
    """
    This bot looks at all legal moves it can make and randomly chooses a move
    that results in a win. If no so move exists a random move is selected.
    """
    is_trainable = False
    
    def __init__(self, filter_losing_moves = False):
        self.filter_losing_moves = filter_losing_moves
    
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
        if winning_moves:
            return Act.play(random.choice(winning_moves))
        
        if self.filter_losing_moves:
            candidates = self.remove_losing_moves(candidates, game_state)
        
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
    
    def remove_losing_moves(self, moves, game_state):
        """
        Returns a copy of the given list of oves but with the losing moves
        removed. If all moves are losing moves then the given list is returned.
        """
        f = lambda m : self.is_not_losing_move(m, game_state)
        filtered_moves = filter(f, moves)
        filtered_moves = list(filtered_moves)
        return filtered_moves if filtered_moves else moves
    
    def is_not_losing_move(self, move, game_state):
        """
        Returns False if there is a move following the given move which wins
        the game. Otherwise returns True.
        """
        # Get the game state the given moves creates and get the legal follow
        # up moves.
        next_state = game_state.copy()
        next_state.take_turn_with_no_checks(Act.play(move))
        candidates = next_state.legal_moves()
        
        # Check if any of the follow up moves end the game.
        for m in candidates:
            next_next_state = next_state.copy()
            next_next_state.take_turn_with_no_checks(Act.play(m))
            if not next_next_state.winner == 0:
                return False
            
        return True
            
        