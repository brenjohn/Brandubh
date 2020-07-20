#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:49:08 2020

@author: john
"""

import numpy as np
import copy

from bots.four_plane_encoder import FourPlaneEncoder
from brandubh import Act, GameState
from bots.random_bot import RandomBot
from keras.models import load_model



class ZeroBot:
    """
    The ZeroBot uses the policy used by alphaGoZero to select moves. Namely,
    if uses a type of Monte Carlo tree search which has been integrated with
    a neural network that evaulates board positions and predicts which
    branches of the search tree will be visted most often.
    
    Attributes:
        num_rounds - The number of nodes that will be added to the tree 
                     structure created when selecting a move. For larger 
                     num_rounds, the bot will take longer to choose a move but
                     it should also pick stringer moves.
                     
        c          - A parameter to balance exploration and exploitation. The
                     bot will explore more for larger c.
                     
        alpha      - The dirichlet noise parameter. The noise is used to
                     increase the probabilty random moves get explored during
                     move selection.
                     
        evaluation_history_old - An array holding results of evaluations of
                                 the bot against an older version of the bot.
                                 
        evaluation_history_ran - An array holding results of evaluations of
                                 the bot against a bot that makes random moves.
                                 
        loss_history           - A list holding the values of the loss 
                                 functions (total loss, soft max - policy head,
                                 mse - value head) during training.
                                 
        encoder    - An encoder object to convert board positions into neural
                     network input and network output to a distribution over
                     moves.
                     
        rand_bot   - A bot that makes random moves. Used for evaluating bot
                     performance after training.
    """
    
    def __init__(self, num_rounds=10, model=None):
        self.num_rounds = num_rounds
        self.c = 2.0
        self.alpha = 0.03
        self.evaluation_history_old = []
        self.evaluation_history_ran = []
        self.loss_history = []
        self.encoder = FourPlaneEncoder()
        self.rand_bot = RandomBot()
        
        if model:
            self.model = model
    
    def select_move(self, game_state, return_visit_counts=False):
        """
        Select a move to make from the given board position (game_state).
        
        The algorithm uses a combination of a neural network and a Monte Carlo
        tree search to search the game tree stemming from the given board
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
        
        # Start with a tree consisting of a root node only. The root node
        # is associated with the given board position.
        root = self.create_node(game_state)
        
        # If no legal moves can be made from the given board position, pass 
        # the turn. This happens when all of the players pieces are surrounded,
        # if the player has no pieces left or if the game is over. 
        if not root.branches:
            if return_visit_counts:
                return Act.pass_turn(), np.zeros(96)
            return Act.pass_turn()
        
        for i in range(self.num_rounds):
            # On each iteration, walk down the tree to a leaf node and select
            # a move to make from the corresponding leaf game state.
            node = root
            next_move = self.select_branch(node)
            while node.has_child(next_move):
                node = node.get_child(next_move)
                next_move = self.select_branch(node)
             
            # Create a new tree node for the selected move and add it to
            # the tree. If the leaf node corresponds to a finished game
            # then don't create a new node and assign a value to the node
            # based on who won.
            if node.state.is_not_over():
                new_state = copy.deepcopy(node.state)
                new_state.take_turn_with_no_checks(Act.play(next_move))
                child_node = self.create_node(new_state, parent=node)
                move = next_move
                value = -1 * child_node.value 
            else:
                move = node.last_move
                # Value of finished board position is 1 if current player 
                # is winner and -1 otherwise.
                value = -1 * node.state.winner * node.state.player
                node = node.parent
            
            # Update the nodes traversed to get to the leaf node with the 
            # new new values induced by the new move.
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value *= -1
            
        # Get the visit counts of the branches if they were requested.
        # TODO: Clean this up
        if return_visit_counts:
            ordered_moves = [self.encoder.decode_move_index(game_state, index)
                             for index in range(96)]
            visit_counts = [root.visit_count(move) 
                            for move in ordered_moves]
                
        # Get a list of possible moves sorted according to visit count,
        # with the move with the highest visit count appearing first.
        moves = [move for move in root.moves()]
        moves = sorted(moves, key=root.visit_count, reverse=True)
        
        # Loop through the sorted moves and return the first legal one.
        for move in moves:
            if not game_state.is_move_illegal(move):
                if return_visit_counts:
                    return Act.play(move), visit_counts
                return Act.play(move)
        
        # If no legal move is found then pass the turn.
        if return_visit_counts:
            return Act.pass_turn(), visit_counts
        return Act.pass_turn()
                
    def create_node(self, game_state, move=None, parent=None):
        """
        This method creates a tree node for the given board position and adds
        it to the tree structure. It will be linked to the given parent node
        and the given move is stored as the last move taken to produce the
        given game state. This is useful for updating the tree structure 
        when other nodes are added to it.
        """
        # Pass the game state to the neural network to both evaluate the 
        # how good the board position is and get the prior probability
        # distribution over possible next moves.
        state_tensor = self.encoder.encode(game_state)
        model_input = state_tensor.reshape(1,7,7,4)
        priors, value = self.model.predict(model_input)
        priors, value = priors[0], value[0][0]
        
        # If a root node is being created, then added some dirichlet noise
        # to the prior probabilities to help exploration.
        if parent == None:
            dirichlet_noise = np.random.dirichlet([self.alpha]*96)
            priors = (priors + dirichlet_noise)/2
        
        # Put the prior probabilities into a dictionary with tuples 
        # representing the aoosciated moves.
        move_priors = {self.encoder.decode_move_index(game_state, idx): prior
                       for idx, prior in enumerate(priors)}
        
        # Create the node for the given game state and attach it to the tree.
        new_node = TreeNode(game_state, value, move_priors, parent, move)
        if parent is not None:
            parent.add_child(move, new_node)
        return new_node
    
    def select_branch(self, node):
        """
        This method selects a move/branch stemming from the given node by 
        picking the move that maximises the following score:
            
            Q + c*p*sqrt(N)/(1+n),
            
        where Q = the estimated expected reward for the move,
              c = a constant balancing exploration-exploitation,
              p = prior probability for the move,
              N = The total number of visits to the given node
              n = the number of those visits that went to the branch 
                  associated with the move
        """
        total_n = node.total_visit_count
        
        def branch_score(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            return q + self.c * p * np.sqrt(total_n)/(1+n)
        
        return max(node.moves(), key=branch_score)
    
    def evaluate_against_bot(self, opponent_bot, num_games):
        zero_bot_player = 1
        score = 0
        num_games_won_as_black = 0
        num_games_won_as_white = 0
        
        for i in range(num_games):
            print('\rPlaying game {0}'.format(i),end='')
            game = GameState.new_game()
            
            max_num_of_turns = 1000
            turns_taken = 0
            
            while game.is_not_over() and turns_taken < max_num_of_turns:
                if game.player == zero_bot_player:
                    action = self.select_move(game)
                else:
                    action = opponent_bot.select_move(game)
                    
                game.take_turn_with_no_checks(action)
                turns_taken += 1
             
                
            if turns_taken < max_num_of_turns:
                score += zero_bot_player*game.winner
            
                if zero_bot_player == game.winner:
                    if zero_bot_player == 1:
                        num_games_won_as_white += 1
                    else:
                        num_games_won_as_black += 1
                        
                zero_bot_player *= -1
                        
            else:
                score -= 1
                zero_bot_player *= -1
                
        print(' done')
        # Save the evaluation score of the bot along with fraction of games
        # won as black/white, the total number of games and the number of
        # epochs the bot has trained for before being evaluated.
        return [score/num_games, 2*num_games_won_as_white/num_games,
                2*num_games_won_as_black/num_games, 
                num_games, len(self.loss_history)]
    
    def evaluate_against_rand_bot(self, num_games):
        print('Evaluating against random bot')
        results = self.evaluate_against_bot(self.rand_bot, num_games)
        self.evaluation_history_ran.append(results)
        
    def evaluate_against_old_bot(self, num_games, 
                                 prefix="model_data/old_bot/"):
        print('Evaluating against old bot')
        old_bot = ZeroBot(1)
        old_bot.load_old_bot(prefix)
        results = self.evaluate_against_bot(old_bot, num_games)
        self.evaluation_history_old.append(results)
        
    def save_losses(self, loss_history):
        losses = [loss[0] for loss in loss_history.history.values()]
        self.loss_history.append(losses)
    
    def save_bot(self, prefix="model_data/"):
        self.model.save(prefix + 'zero_model.h5')
        
        attributes = {"num_rounds" : self.num_rounds,
                      "c" : self.c,
                      "alpha" : self.alpha,
                      "loss_history" : self.loss_history,
                      "evaluation_history_old" : self.evaluation_history_old,
                      "evaluation_history_ran" : self.evaluation_history_ran}
        
        np.save(prefix + "model_attributes.npy", attributes)
        
    def load_bot(self, prefix="model_data/"):
        self.model = load_model(prefix + 'zero_model.h5')
        attributes = np.load(prefix + "model_attributes.npy",
                             allow_pickle='TRUE').item()
        
        self.num_rounds = attributes["num_rounds"]
        self.c = attributes["c"]
        self.alpha = attributes["alpha"]
        if "loss_history" in attributes:
            self.loss_history = attributes["loss_history"]
        self.evaluation_history_old = attributes["evaluation_history_old"]
        self.evaluation_history_ran = attributes["evaluation_history_ran"]
        
    def save_as_old_bot(self, prefix="model_data/old_bot/"):
        self.save_bot(prefix)
        
    def load_old_bot(self, prefix="model_data/old_bot/"):
        self.load_bot(prefix)



class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0 # when divided by visit count should give the
                             # average value of the move corresponding to this
                             # branch

        
        
class TreeNode:
    def __init__(self, game_state, value, priors, parent, last_move):
        self.state = game_state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        
        # Used when branch stemming from this node is being selected.
        self.total_visit_count = 0 # ?? sum branch visits?
        
        self.branches = {}
        for move, prior in priors.items():
            if not game_state.is_move_illegal(move):
                self.branches[move] = Branch(prior)
                
        self.children = {}
         
    def moves(self):
        return self.branches.keys()
    
    def add_child(self, move, child_node):
        self.children[move] = child_node
        
    def has_child(self, move):
        return move in self.children
    
    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0
        return branch.total_value / branch.visit_count
    
    def prior(self, move):
        return self.branches[move].prior
    
    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
    
    def record_visit(self, move, value):
        self.total_visit_count += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value