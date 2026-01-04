#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:36:55 2026

@author: john
"""

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
    def __init__(self):
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
        next_move = node.next_move()
        
        while node.has_child(next_move):
            node.increment_virtual_loss(next_move)
            node = node.get_child(next_move)
            self.node = node
            next_move = node.next_move()
            
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