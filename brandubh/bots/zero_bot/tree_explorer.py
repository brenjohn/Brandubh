#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:36:55 2026

@author: john
"""

import numpy as np

from ...game import Act
from .search_tree import TreeNode


class TreeExplorer:
    """
    TreeExplorer objects are responsible for traversing a search tree according 
    to the PUCT (polynomial upper confidence tree) rule.
    
    A virtual loss, stored in the branch objects, is used to modify the PUCT
    score of branches traversed by a tree_explorer to discourage other 
    TreeExplorer objects from exploring the same branch. This 
    facilitates concurrent exploration of the search tree and batching of 
    game states to be processed by the neural network network.
    
    Expanding a leaf node of a search tree happens in the following steps:
        
        1 - A tree_explorer initialised with the root node uses the climb_down
        method to traverse the tree to a leaf node. The virtual loss of 
        traversed branches is increased during this process.
        
        2 - The leaf node is expanded with the expand_branch method
        
        3 - The tree_ecplorer then traverses back to the root node using the
        climb_up method which also updates search statistics stored in the 
        branches along the way with the appropriate values. Changes to the 
        virtual loss are undone alos during this process.
    """
    def __init__(self, c):
        self.c = c             # Constant in the PUCT formula.
        self.node      = None  # The current node the explorer is on.
        self.value     = None  # The value to be used update the parent branch.
        self.next_move = None  # The next move selected by the PUCT score.
    
    
    def set_node(self, node):
        self.node = node
    
    
    def get_next_state(self):
        """Returns the game state created by the stored next move. The next 
        move is set by the climb down method.
        """
        if self.next_move is not None:
            action = Act.play(self.next_move)
        else:
            # If the current player can't make any moves then next move will be
            # 'None', meaning the player passes the turn.
            action = Act.pass_turn()
        next_state = self.node.state.copy()
        next_state.take_turn_with_no_checks(action)
        return next_state
    
    
    def expand_branch(self, state, priors, value):
        """Creates a new node for the given state and adds it to the tree.
        """
        # Create the node for the given game state, with the predicted value
        # and priors, and attach it to the tree.
        new_node = TreeNode(state, value, priors, self.node, self.next_move)
        self.node.add_child(self.next_move, new_node)
        self.value = -1 * value
    
    
    def evaluate_terminal_leaf(self):
        # If the current game state is over, then the last
        # player must have won the game. Thus the value/reward for the
        # other player is 1. The current node is not updated with
        # the new reward as no branches can stem from a finished game
        # state. 
        self.value = 1
    
    
    def climb_down(self):
        """climb up the tree to a leaf node and select a move to make from the 
        corresponding leaf game state.
        """
        node = self.node
        next_move = self.select_branch()
        
        while node.has_child(next_move):
            node.increment_virtual_loss(next_move)
            node = node.get_child(next_move)
            self.node = node
            next_move = self.select_branch()
            
        node.lock_branch(next_move)
        self.next_move = next_move
    
    
    def climb_up(self):
        """Climb down the tree and update the nodes traversed to get to the 
        leaf node with the new value for the new move.
        """
        node = self.node
        move = self.next_move
        value = self.value
        
        node.unlock_branch(move)
        while node.parent is not None:
            node.record_visit(move, value)
            move = node.last_move
            node = node.parent
            node.decrement_virtual_loss(move)
            value *= -1
            
        node.record_visit(move, value)
        self.node = node
    
    
    def select_branch(self):
        """
        This method selects a move/branch stemming from the given node by 
        picking the move that maximises the following PUCT score:
            
            Q + c * p * sqrt(N) / (1+n),
            
        where Q = the estimated expected reward for the move,
              c = a constant balancing exploration-exploitation,
              p = prior probability for the move,
              N = The total number of visits to the given node
              n = the number of those visits that went to the branch 
                  associated with the move
                  
        Christopher D. Rosin - Multi-armed Bandits with Episode Context
        """
        self.c_sqrt_total_n = np.sqrt(self.node.total_visit_count) * self.c
        
        moves = self.node.moves()
        if moves:
            return max(moves, key=self.branch_score)
        else:
            # If moves is empty then no legal moves can be made from the game
            # state corresponding to the given node.
            return None
    
    
    def branch_score(self, move):
        q, p, n = self.node.branch_score_stats(move)
        return q + p * self.c_sqrt_total_n/(1+n)
        