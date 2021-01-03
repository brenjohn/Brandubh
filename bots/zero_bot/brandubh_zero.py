#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:49:08 2020

@author: john

This file contains classes used to create a bot that can play brandubh using
the AlphaGo-Zero approach.
"""

import numpy as np
import copy

from brandubh import Act, GameState
from bots.random_bot import RandomBot
from bots.zero_bot.zero_training_utils import random_starting_position



class ZeroBot:
    """
    The ZeroBot uses the policy used by AlphaGoZero to select moves. Namely,
    it uses a type of Monte Carlo tree search which has been integrated with
    a neural network that evaulates board positions and predicts which
    branches of the search tree will be visted most often. The move it selects
    is the move corresponding to the most visited branch during this tree 
    search.
    
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
                     
        rand_bot   - A bot that makes random moves. Used for evaluating bot
                     performance after training.
    """
    
    def __init__(self, num_rounds=10, network=None):
        self.num_rounds = num_rounds
        self.c = 2.0
        self.alpha = 0.03
        self.evaluation_history_old = []
        self.evaluation_history_ran = []
        self.loss_history = []
        self.rand_bot = RandomBot()
        
        if network:
            self.network = network
    
    def select_move(self, game_state, return_visit_counts=False):
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
        
        # Start with a tree consisting of a root node only. The root node
        # is associated with the given board position.
        root = self.create_node(game_state)
        
        # If no legal moves can be made from the given board position, pass 
        # the turn. This happens when all of the players pieces are surrounded,
        # if the player has no pieces left or if the game is over. 
        if not root.branches:
            if return_visit_counts:
                return Act.pass_turn(), {}
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
                if next_move:
                    new_state = copy.deepcopy(node.state)
                    new_state.take_turn_with_no_checks(Act.play(next_move))
                    child_node = self.create_node(new_state, 
                                                  move=next_move, parent=node)
                    move = next_move
                    value = -1 * child_node.value 
                else:
                    # If the current player can't make any moves from the
                    # selected gamestate then next_move will be 'None' meaning
                    # the player passes the turn.
                    new_state = copy.deepcopy(node.state)
                    new_state.take_turn_with_no_checks(Act.pass_turn())
                    child_node = self.create_node(new_state, 
                                                  move=next_move, parent=node)
                    move = next_move
                    value = -1 * child_node.value
            else:
                # If the game in the current state is over, then the last
                # player must have won the game. Thus the value/reward for the
                # other player is 1. The current node is not updated with
                # the new reward as no branches can stem from a finished game
                # state.
                move = node.last_move
                node = node.parent
                value = 1
            
            # Update the nodes traversed to get to the leaf node with the 
            # new value for the new move.
            while node is not None:
                node.record_visit(move, value)
                move = node.last_move
                node = node.parent
                value *= -1
            
        # Get the visit counts of the branches if they were requested.
        if return_visit_counts:
            visit_counts = {}
            for move in root.branches.keys():
                visit_counts[move] = root.branches[move].visit_count
                
        # Get a list of possible moves sorted according to visit count,
        # the move with the highest visit count should be first in the list.
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
        given game state. This is useful for trversing and updating the tree 
        structure when other nodes are added to it.
        """
        # Pass the game state to the neural network to both evaluate the 
        # how good the board position is and get the prior probability 
        # distribution over possible next moves (ie the predicted distribution 
        # of visit counts).
        move_priors, value = self.network.predict(game_state)
        
        # If a root node is being created, then add some dirichlet noise
        # to the prior probabilities to help exploration.
        if parent == None:
            dirichlet_noise = np.random.dirichlet([self.alpha]*96)
            for (i, move) in enumerate(move_priors.keys()):
                move_priors[move] = (move_priors[move] + dirichlet_noise[i])/2
        
        # Create the node for the given game state, with the predicted value
        # and priors, and attach it to the tree.
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
        
        moves = node.moves()
        if moves:
            return max(moves, key=branch_score)
        else:
            # If moves is empty then no legal moves can be made from the game
            # state corresponding to the given node.
            return None
    
    def evaluate_against_bot(self, opponent_bot, num_games, 
                             num_white_pieces = None, 
                             num_black_pieces = None,
                             max_num_of_turns = 1000):
        """
        This method evaluates the current bot against a given opponent bot
        by letting them play a number of games against each other. The number
        of games played is specified by 'num_games'. A random starting 
        position for the games is generated if a maximum number of white and
        black pieces is given by the parameters 'num_white_pieces' and
        'num_black_pieces', otherwise the regular starting position is used.
        
        If the number of turns taken in a game exceeds the given maximum, then
        the game ends and drawn up as a win for the opponent bot.
        """
        zero_bot_player = 1
        score = 0
        num_games_won_as_black = 0
        num_games_won_as_white = 0
        
        # Play 'num_games' games of brandubh
        for i in range(num_games):
            print('\rPlaying game {0}, score: w = {1}, b = {2}.'.format(i, 
                    num_games_won_as_white, num_games_won_as_black),end='')
            
            # If a maximum number of white or black pieces is given, then
            # use a random starting position for the game.
            if num_white_pieces or num_black_pieces:
                starting_board = random_starting_position(num_white_pieces, 
                                                          num_black_pieces)
                game = GameState.new_game(starting_board)
            else:
                game = GameState.new_game()
            
            # Get both bots to play a game of brandubh.
            turns_taken = 0
            while game.is_not_over() and turns_taken < max_num_of_turns:
                if game.player == zero_bot_player:
                    action = self.select_move(game)
                else:
                    action = opponent_bot.select_move(game)  
                game.take_turn_with_no_checks(action)
                turns_taken += 1
             
                
            # At the end of the game, increment counts keeping track of how
            # many games the current bot won against the opponent bot and 
            # get the bots to switch sides for the next game.
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
                
        print(' done.')
        # Return the evaluation score of the bot along with fraction of games
        # won as black/white, the total number of games and the number of
        # epochs the bot has trained for before being evaluated.
        return [score/num_games, 2*num_games_won_as_white/num_games,
                2*num_games_won_as_black/num_games, 
                num_games, len(self.loss_history)]
    
    def evaluate_against_rand_bot(self, num_games, 
                                  num_white_pieces = None, 
                                  num_black_pieces = None):
        """
        Function to evaluate how good the current bot is against a bot who
        makes random moves.
        """
        print('Evaluating against random bot')
        results = self.evaluate_against_bot(self.rand_bot, num_games,
                                            num_white_pieces, 
                                            num_black_pieces)
        self.evaluation_history_ran.append(results)
        
    def evaluate_against_old_bot(self, num_games,
                                 num_white_pieces = None, 
                                 num_black_pieces = None,
                                 prefix="model_data/old_bot/"):
        """
        Function to evaluate how good the current bot is against an older 
        version of the current bot whoes weights are save under the directory
        given by the parameter 'prefix'. 
        """
        print('Evaluating against old bot')
        old_bot = ZeroBot(1)
        old_bot.load_old_bot(prefix)
        results = self.evaluate_against_bot(old_bot, num_games,
                                            num_white_pieces, 
                                            num_black_pieces)
        self.evaluation_history_old.append(results)
        
    def save_losses(self, loss_history):
        """
        Method to save the evaulations of the loss function of the neural
        network on training data.
        """
        losses = [loss[0] for loss in loss_history.history.values()]
        self.loss_history.append(losses)
    
    def save_bot(self, prefix="model_data/"):
        """
        Method to save the attributes of the current bot and the weights of
        its neural network under the directory given by the parameter 
        'prefix'
        """
        network_load_command = self.network.save_network(prefix)
        
        attributes = {"num_rounds" : self.num_rounds,
                      "c" : self.c,
                      "alpha" : self.alpha,
                      "loss_history" : self.loss_history,
                      "evaluation_history_old" : self.evaluation_history_old,
                      "evaluation_history_ran" : self.evaluation_history_ran,
                      "network_load_command": network_load_command}
        
        np.save(prefix + "model_attributes.npy", attributes)
        
    def load_bot(self, prefix="model_data/"):
        """
        Method to load the attributes and neural network saved under the given
        directory.
        """
        attributes = np.load(prefix + "model_attributes.npy",
                             allow_pickle='TRUE').item()
        
        self.num_rounds = attributes["num_rounds"]
        self.c = attributes["c"]
        self.alpha = attributes["alpha"]
        if "loss_history" in attributes:
            self.loss_history = attributes["loss_history"]
        self.evaluation_history_old = attributes["evaluation_history_old"]
        self.evaluation_history_ran = attributes["evaluation_history_ran"]
        
        network_load_command = attributes["network_load_command"]
        exec(network_load_command)
        self.network.load_network(prefix)
        
    def save_as_old_bot(self, prefix="model_data/old_bot/"):
        """
        Method to save the current bot as the 'old_bot' used in bot evaluation.
        """
        self.save_bot(prefix)
        
    def load_old_bot(self, prefix="model_data/old_bot/"):
        """
        Method to load the old_bot for evaluating the current bot.
        """
        self.load_bot(prefix)



class Branch:
    """
    Instances of this class are used to store statistics gathered, by the
    ZeroBot select_move algorithm, on how often a branch of the decision tree
    stemming from a particular gamestae was visited and the estimated value
    of the resulting board position. It also saves the prior probability of
    the move associated with an instance of Branch as predicted by the neural
    network.
    """
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0 # when divided by visit count should give the
                             # average value of the board corresponding to this
                             # branch

        
        
class TreeNode:
    """
    This class can represent a node (corresponding to a game state) in the 
    decision tree stemming from a game state of brandubh.
    
    Instances of this class are used to build a tree structure to record the
    search history of the ZeroBot select_move algorithm. It saves an instance
    of the game state it represents, the expected value of that game state as
    predicted by the neural network, a reference to its parent node if it has
    one and a tuple representing the previous move of the game corresponding 
    game state which created it.
    
    It also contains two dictionaries, indexed by game moves, which hold 
    references to any child nodes attached to the current instance in the tree
    structure and branch objects containing statistics regarding the search 
    history of the select_move method.
    
    TODO: It may be cleaner to combine the TreeNode and Branch classes into
    one. Only a single dictionary would be required then. Should think about
    if or how this would affect performance of the select_move method.
    """
    def __init__(self, game_state, value, priors, parent, last_move):
        self.state = game_state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        
        # Used when branch stemming from this node is being selected.
        self.total_visit_count = 0
        
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
    
    def get_child(self, move):
        return self.children[move]
    
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
        # If the move isn't a pass
        if move:
            self.branches[move].visit_count += 1
            self.branches[move].total_value += value