#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:49:08 2020

@author: john

This file contains classes used to create a bot that can play brandubh using
the AlphaGo-Zero approach.
"""
import numpy as np
import os

from ...game import Act
from .zero_network import ZeroNet
from .search_tree import TreeNode
from .tree_explorer import TreeExplorer


class ZeroBot:
    """
    The ZeroBot uses the policy used by AlphaGoZero to select moves. Namely,
    it uses a type of Monte Carlo tree search which has been integrated with
    a neural network that evaulates board positions and predicts which
    branches of the search tree will be visted most often. The move it selects
    is the move corresponding to the most visited branch during this tree 
    search.
    
    Arguments:
        evals_per_turn : 
            The number of nodes that will be added to the tree structure when 
            selecting a move. For larger num_rounds, the bot will take longer 
            to choose a move but it should also pick stringer moves.
            
        batch_size :
            The number of tree nodes to evaluate at any one time.
        
        c_puct :
            A parameter to balance exploration and exploitation. The bot will 
            explore more for larger c_puct.
                     
        alpha :
            The dirichlet noise parameter. The noise is used to increase the 
            probabilty random moves get explored during move selection.
                     
        sampling_turns :
            The number of turns at the start of a game where moves will be
            sampled from a probability distribution defined by the move visit
            count statitics in the tree. After this number of turns, the move
            with the largest visit count is selected.
            
        network :
            The neural network to use.
    """
    
    def __init__(
            self, 
            evals_per_turn = 7000, 
            batch_size     = 70,
            c_puct         = 1.4,
            alpha          = 0.15,
            sampling_turns = 6,
            network        = None
        ):
        self.evals_per_turn = evals_per_turn
        self.batch_size     = batch_size
        self.c_puct         = c_puct
        self.alpha          = alpha
        self.sampling_turns = sampling_turns
        
        if network:
            self.network = network
        else:
            self.network = ZeroNet()
                
        TreeNode.c_puct = c_puct
        self.tree_explorers = [TreeExplorer() for i in range(batch_size)]
        self.root = None
        self.compile_network((1.0, 0.1))
    
    
    def select_move(
            self, 
            game_state,
            reuse_search_tree=True
        ):
        """
        Select a move to make from the given board position (game_state).
        
        The algorithm uses a combination of a neural network and a Monte Carlo
        tree search to search the decision tree stemming from the given board
        position. It returns the move associated with the most visited branch 
        stemming from the root.
        
        This method creates a tree structure representing the search history
        of the algorithm and is used to save evaluations of board positions
        and statistics regarding nodes visited.
        
        If return_visit_counts is true, the distribution of visits over the 
        branches of the root in the search tree will be returned along with
        the selected move. This distribution can be used to train the neural
        network.
        """
        
        # If a search tree is already saved, reuse the subtree relevant to
        # the given game state. Otherwise, start with a tree consisting of a 
        # root node only. The root node is associated with the given board 
        # position.
        if reuse_search_tree:
            self.update_root_to_current_game_state(game_state)
            
        if self.root is None:
            self.root = self.create_root_node(game_state.copy())
        self.root.add_noise(self.alpha)
        
        
        # If no legal moves can be made from the given board position, pass 
        # the turn. This happens when all of the players pieces are surrounded,
        # if the player has no pieces left or if the game is over. 
        if not self.root.branches:
            return Act.pass_turn()
        
        # Run the hybrid neural network - Monte Carlo tree search algorithm to
        # update the current search tree with new board evaluations.
        self.update_tree()
        
        # Select one of the the possible moves using visit count statistics
        # from the tree.
        act = self._select_move(num_turns = game_state.num_moves)
        return act
    
    
    def _select_move(self, num_turns):
        """
        Creates a list of possible moves and selects a one with one of the 
        following methods:
        
        1) If the number of turns in the game is less than a certain 
        threshold, the move is randomly sampled from the prob dist defined by 
        the visit counts.
        
        2) Otherwise, the move with the highest visit count is selected.
        """
        moves = [move for move in self.root.moves()]
        if moves:
            if num_turns < self.sampling_turns:
                move_distribution = np.asarray([
                    self.root.branches[move].visit_count for move in moves
                ])
                move_distribution = move_distribution / sum(move_distribution)
                move_ind = np.random.choice(len(moves), p=move_distribution)
                move = moves[move_ind]
            else:
                move = max(moves, key=self.root.visit_count)
            
            # Return the move as an Act.
            return Act.play(move)
        
        # If no legal move is found then pass the turn.
        return Act.pass_turn()
        
    
    def update_root_to_current_game_state(self, game_state):
        """
        Attempts to reuse the existing search tree by finding the current 
        game_state within the tree's descendants.
        """
        # If a search tree is saved.
        if self.root is None:
            return
        
        # Collect the moves made since the last turn taken.
        moves_since_last_turn = []
        historic_state = game_state.history
        root_found = False
        
        for _ in range(3):
            if self.root.corresponds_to(historic_state):
                root_found = True
                break
            if historic_state.previous_state is not None:
                moves_since_last_turn.insert(0, historic_state.last_move)
                historic_state = historic_state.previous_state
            else:
                break
        
        # Set the root to None if it isn't in the local history.
        if not root_found:
            self.root = None
            return
        
        # Update the root with the moves made since the last turn.
        for move in moves_since_last_turn:
            if self.root.has_child(move):
                self.root = self.root.get_child(move)
            else:
                # Set root to none if a relevant subtree doesn't exist.
                self.root = None
                return
            
        # Disconnect the root from any parent it might have.
        if self.root is not None:
            self.root.parent = None
    
    
    def update_tree(self):
        """
        Runs the hybrid Monte Carlo - neural network tree search to populate
        the search tree with board evaluations.
        """
        states_buffer = [None] * len(self.tree_explorers)
        for explorer in self.tree_explorers:
            explorer.set_node(self.root)
        
        evals_made = 0
        while evals_made != self.evals_per_turn:
            
            explorers_ready = 0
            while (
                    explorers_ready < self.batch_size and 
                    evals_made < self.evals_per_turn
                ):
                
                explorer = self.tree_explorers[explorers_ready]
                explorer.climb_down()
                
                next_state = explorer.get_next_state()
                if next_state.is_not_over():
                    states_buffer[explorers_ready] = next_state
                    explorers_ready += 1
                else:
                    explorer.evaluate_terminal_leaf()
                    explorer.climb_up()
                    
                evals_made += 1
                    
            explorers = self.tree_explorers[0:explorers_ready]
            states_to_be_added = states_buffer[0:explorers_ready]
            if explorers_ready > 0:
                predictions = self.network.predict(states_to_be_added)
                
                new_node_args = zip(states_to_be_added, predictions, explorers)
                for state, prediction, explorer in new_node_args:
                    explorer.expand_branch(state, *prediction)
                    explorer.climb_up()

                
    def create_root_node(self, game_state):
        """
        This method creates a tree node for the given board position and adds
        it to the tree structure. It will be linked to the given parent node
        and the given move is stored as the last move taken to produce the
        given game state. This is useful for trversing and updating the tree 
        structure when other nodes are added to it.
        """
        # Pass the game state to the neural network to both evaluate the 
        # how good the board position is and get the prior probability 
        # distribution over possible next moves (ie the predicted distribution 
        # of visit counts).
        prediction = self.network.predict([game_state])
        move_priors, value = prediction[0]
        
        # Create the node for the given game state, with the predicted value
        # and priors, and attach it to the tree.
        return TreeNode(game_state, value, move_priors, None, None)
        
        
    def turn_on_eval_mode(self, look_ahead = None, **kwargs):
        """
        """
        self.old_alpha = self.alpha
        self.old_evals_per_turn = self.evals_per_turn
        
        self.alpha = 0.0
        if look_ahead is not None:
            self.evals_per_turn = look_ahead
    
    
    def turn_off_eval_mode(self):
        """
        """
        self.alpha = self.old_alpha
        self.evals_per_turn = self.old_evals_per_turn
    
    
    def save_bot(self, prefix="model_data/"):
        """
        Method to save the attributes of the current bot and the weights of
        its neural network under the directory given by the parameter 
        'prefix'
        """
        if not os.path.exists(prefix):
            os.makedirs(prefix)
            
        network_load_command = self.network.save_network(prefix)
        attributes = {
            "evals_per_turn" : self.evals_per_turn,
            "c_puct"         : self.c_puct,
            "batch_size"     : self.batch_size,
            "alpha"          : self.alpha,
            "sampling_turns" : self.sampling_turns,
            "network_load_command": network_load_command
        }
        
        np.save(prefix + "model_attributes.npy", attributes)
        
    
    def load_bot(self, prefix="model_data/"):
        """
        Method to load the attributes and neural network saved under the given
        directory.
        """
        attributes = np.load(prefix + "model_attributes.npy",
                             allow_pickle='TRUE').item()
        
        self.evals_per_turn = attributes["evals_per_turn"]
        self.c_puct = attributes["c_puct"]
        self.batch_size = attributes["batch_size"]
        self.tree_explorers = [TreeExplorer() for i in range(self.batch_size)]
        self.alpha = attributes["alpha"]
        
        network_load_command = attributes["network_load_command"]
        exec(network_load_command)
        self.network.load_network(prefix)
    
        
    def compile_network(self, loss_weights):
        self.network.compile_network(*loss_weights)
    
    
    def get_encoder(self):
        """Returns the encoder object used to encode board states as tensors 
        for the neural network and creating training data from self play games.
        """
        return self.network.encoder