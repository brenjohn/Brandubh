#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 12:44:34 2026

@author: john

This submodule defines the TreeNode class, representing a node in the ZeroBot
search tree, and a Branch class for holding tree search statistics.
"""

import numpy as np


class Branch:
    """Branch class for storing statistics gathered by the ZeroBot algorithm.
    
    Tracked statistics are:
        prior        - The prior probability for the branch.
        
        visit count  - The number of times the branch has been explored.
        
        virtual loss - A temporary virtual loss value used for concurrent 
                       branch exploration (see TreeExplorer class)
        
        total value  - Sum of all values of descendant nodes of this branch.
    """
    def __init__(self, prior):
        self.prior        = prior
        self.visit_count  = 0
        self.virtual_loss = 0
        self.total_value  = 0
    
    
    def expected_value(self):
        """Returns the estimated value of the node stemming from this branch.
        """
        expected_value = 0
        if self.visit_count > 0:
            expected_value = (self.total_value + self.virtual_loss)
            expected_value /= self.visit_count
        return expected_value
    
    
    def search_stats(self):
        """Returns the expected value, prior and visit count for the branch.
        """
        return self.expected_value(), self.prior, self.visit_count

        
        
class TreeNode:
    """
    This class can represent a node (corresponding to a game state) in the 
    decision/search tree used in the ZeroBot algorithm.
    
    Instances of this class are used to build a tree structure to record the
    search history of the ZeroBot select_move algorithm. It saves an instance
    of the game state it represents, the expected value of that game state as
    predicted by the neural network, a reference to its parent node if it has
    one and a tuple representing the previous move made in the game that 
    created the current game state.
    
    It also contains two dictionaries, indexed by game moves, which hold 
    references to any child nodes, attached to the current instance in the tree
    structure, and branch objects containing statistics regarding the search 
    history of the select_move method.
    
    The PUCT rule is used to select moves which uses a constant c_puct (saved
    as a class attribute) for balancing exploration and exploitation. The
    ZeroBot sets this class variable before any instances of TreeNode are 
    created.
    """
    c_puct = 1  # Should be manually set before any instances are created.
    
    def __init__(self, game_state, predicted_value, priors, parent, last_move):
        self.state      = game_state
        self.pred_value = predicted_value
        self.parent     = parent
        self.last_move  = last_move  # The move that created the current state. 
        
        self.total_visit_count = 1  # The creation of a node counts as a visit.
        self.c_sqrt_total_n = self.c_puct
        self.children = {}
        self.branches = {
            move : Branch(prior) for move, prior in priors.items()
        }
        
    
    #=========================================================================#
    #                        Getter and setter methods
    #=========================================================================#
    
    def add_child(self, move, child_node):
        self.children[move] = child_node
        
    def has_child(self, move):
        return move in self.children
    
    def get_child(self, move):
        return self.children[move]
    
    def moves(self):
        return self.branches.keys()
    
    def expected_value(self, move):
        return self.branches[move].expected_value()
    
    def predicted_value(self, move):
        pred_value = 0
        if move in self.children:
            pred_value = self.children[move].pred_value
        return pred_value
    
    def branch_search_stats(self, move):
        if move in self.branches:
            return self.branches[move].search_stats()
        return 0, 0, 0
    
    def prior(self, move):
        return self.branches[move].prior
    
    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
    
    def visit_counts(self):
        return {
            move : self.branches[move].visit_count
            for move in self.branches.keys()
        }
    
    def is_not_terminal_leaf(self):
        return self.state.is_not_over()
    
    def branch_puct_score(self, move):
        """Returns the PUCT score for the given move. Assumes c_sqrt_total_n
        is up to date.
        """
        q, p, n = self.branch_search_stats(move)
        return q + p * self.c_sqrt_total_n/(1+n)
    
    #=========================================================================#
    #                        Tree traversal methods
    #=========================================================================#
    
    # The following methods are used by TreeExplorer objects while traversing
    # the search tree to select which move to explore next, update search 
    # statistics, lock branches from being searched by other TreeExplorer 
    # objects and update the virtual loss values of branches.
    
    def next_move(self):
        """Selects a move/branch stemming from this node by picking the move 
        that maximises the following PUCT score:
            
            Q + c * p * sqrt(N) / (1+n),
            
        where Q = the estimated expected reward for the move,
              c = a constant balancing exploration-exploitation,
              p = prior probability for the move,
              N = The total number of visits to the given node
              n = the number of those visits that went to the branch 
                  associated with the move
                  
        Christopher D. Rosin - Multi-armed Bandits with Episode Context
        """
        self.c_sqrt_total_n = np.sqrt(self.total_visit_count) * self.c_puct
        moves = self.moves()
        return max(moves, key=self.branch_puct_score) if moves else None
    
    def increment_virtual_loss(self, move):
        if move is not None:
            branch = self.branches[move]
            branch.virtual_loss -= 1
            branch.visit_count += 1
        
    def decrement_virtual_loss(self, move):
        if move is not None:
            branch = self.branches[move]
            branch.virtual_loss += 1
            branch.visit_count -= 1
    
    def lock_branch(self, move):
        if move is not None:
            # Other PUCT scores will never be this negative
            self.branches[move].virtual_loss = -700
            self.branches[move].visit_count += 1
        
    def unlock_branch(self, move):
        if move is not None:
            self.branches[move].virtual_loss = 0
            self.branches[move].visit_count -= 1
    
    def record_visit(self, move, value):
        self.total_visit_count += 1
        # If the move isn't a pass
        if move is not None:
            self.branches[move].visit_count += 1
            self.branches[move].total_value += value
    
    
    #=========================================================================#
    #                 Utility methods for alphazero algorithm
    #=========================================================================#
    
    def add_noise(self, alpha):
        """Adds Dirichlet noise to the prior distribution over branches.
        """
        if alpha > 0:
            num_branches = len(self.branches)
            noise = np.random.gamma(alpha, 1, num_branches)
            N = sum(noise)
            for i, branch in enumerate(self.branches.values()):
                branch.prior = (0.75 * branch.prior + 0.25 * noise[i] / N)
    
    
    def corresponds_to(self, history_link):
        """Returns True if this TreeNode represents a game state that is
        equivalent to the given historic state.
        """
        if history_link:
            player, game_set = self.state.player, self.state.game_set
            return history_link.corresponds_to(player, game_set)
        return False
    
    
    def get_search_stats(self, moves):
        """Collects search stats from the branches and child nodes stemming 
        from this node. This can be used to monitor tree statitics during a 
        game for inspection/debugging purposes. 
        """
        prior_dist = {}
        visit_dist = {}
        avrg_value = {}
        puct_score = {}
        pred_value = {}
        self.c_sqrt_total_n = np.sqrt(self.total_visit_count) * self.c_puct
        
        for move in moves:
            move_key = str(move)
            value, prior_prob, visit_count = self.branch_search_stats(move)
            prior_dist[move_key] = prior_prob
            visit_dist[move_key] = visit_count / self.total_visit_count
            avrg_value[move_key] = value
            puct_score[move_key] = self.branch_puct_score(move)
            pred_value[move_key] = float(self.predicted_value(move))
            
        return {
            'prior_dist' : prior_dist,
            'visit_dist' : visit_dist,
            'avrg_value' : avrg_value,
            'puct_score' : puct_score,
            'pred_value' : pred_value
        }