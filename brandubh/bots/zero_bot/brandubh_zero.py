#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:49:08 2020

@author: john

This module defines the ZeroBot class, an agent for playing brandubh using the
AlphaGo-Zero approach.
"""

import json
import numpy as np
from pathlib import Path

from ...game import Act
from .search_tree import TreeNode
from .tree_explorer import TreeExplorer
from .zero_network import ZeroNet


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
        
        if network is not None:
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
        position. It returns a move selected using the distribution of visits 
        over branches stemming from the root node (current game state). (See
        _select_move)
        
        This method creates a tree structure representing the search history
        of the algorithm and is used to save evaluations of board positions
        and statistics regarding nodes visited. (See the TreeNode class)
        
        If reuse_search_tree is true, the created tree structure presists
        between turns of the game and will be reused to avoid re-evaluating
        possible future board positions.
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
        """Selects a possible move using one of the following methods:
        
        1) If the number of turns in the game is less than a certain threshold
        (sampling_turns), the move is randomly sampled from the prob dist 
        defined by the distribution of visit counts over next moves.
        
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
        """Attempts to reuse the existing search tree by finding the current 
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
        """Runs the hybrid Monte Carlo - neural network tree search to populate
        the search tree with board evaluations.
        """
        # Prepare tree explorers and a buffer to hold game states to process
        # and add to the search tree.
        states_buffer = [None] * len(self.tree_explorers)
        for explorer in self.tree_explorers:
            explorer.set_node(self.root)
        
        # Continue until `evals_per_turn` nodes have been added to the tree.
        evals_made = 0
        while evals_made != self.evals_per_turn:
            
            # Get all tree explorers to climb down the tree to a leaf node and
            # select a possible next state.
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
            
            # Evaluate the selected states and add them to the tree.
            explorers = self.tree_explorers[0:explorers_ready]
            states_to_be_added = states_buffer[0:explorers_ready]
            if explorers_ready > 0:
                predictions = self.network.predict(states_to_be_added)
                
                new_node_args = zip(states_to_be_added, predictions, explorers)
                for state, prediction, explorer in new_node_args:
                    explorer.expand_branch(state, *prediction)
                    explorer.climb_up()

                
    def create_root_node(self, game_state):
        """Creates a root tree node for the given game state.
        """
        # First pass the game state to the neural network to both evaluate the 
        # how good the board position is and get the prior probability 
        # distribution over possible next moves. Then create the node.
        prediction = self.network.predict([game_state])
        move_priors, value = prediction[0]
        return TreeNode(game_state, value, move_priors, None, None)
        
        
    def turn_on_eval_mode(self, look_ahead = None, **kwargs):
        """Turns of dirichlet noise and adjusts the look ahead for evaluation.
        """
        self.old_alpha = self.alpha
        self.old_evals_per_turn = self.evals_per_turn
        
        self.alpha = 0.0
        if look_ahead is not None:
            self.evals_per_turn = look_ahead
    
    
    def turn_off_eval_mode(self):
        """Returns dirichlet noise and look ahead to what it was before eval
        mode was turned on.
        """
        self.alpha = self.old_alpha
        self.evals_per_turn = self.old_evals_per_turn
    
    
    def save_bot(self, model_dir=Path("model/")):
        """Saves the bot to the given directory.
        """
        model_dir.mkdir(exist_ok=True)
        self.network.save_network(model_dir)
        attributes = {
            "evals_per_turn" : self.evals_per_turn,
            "c_puct"         : self.c_puct,
            "batch_size"     : self.batch_size,
            "alpha"          : self.alpha,
            "sampling_turns" : self.sampling_turns
        }
        with open(model_dir / "zero_bot_attributes.json", 'w') as file:
            json.dump(attributes, file, indent=4)
        
    
    @classmethod
    def load_bot(cls, model_dir=Path("model/")):
        """Loads a bot from the given directory.
        """
        with open(model_dir / "zero_bot_attributes.json", 'r') as file:
            attributes = json.load(file)
        attributes['network'] = ZeroNet.load(model_dir)
        return ZeroBot(**attributes)
    
        
    def compile_network(self, loss_weights):
        self.network.compile_network(*loss_weights)
    
    
    def get_encoder(self):
        """Returns the encoder object used to encode board states as tensors 
        for the neural network and creating training data from self play games.
        """
        return self.network.encoder