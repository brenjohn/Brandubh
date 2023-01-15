#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:28:27 2020

@author: john

This file has some useful functions for gathering games that a ZeroBot
can learn from and creating training data for the associated neural network.
"""
import numpy as np
import random
import os

from ..game import GameState



def simulate_game(white, black, starting_board=None, max_moves=0, eps=0):
    """
    A function to get the provided bots to play a single game of brandubh
    against eachother in order to generate training data for the bots.
    
    Each turn of the game, after a move is selected, the game-state is encoded
    as tensor (the same tensor the neural network takes as input to predict
    the state value and move priors) and appended to a boards list to be
    returned at the end of the game. These will become X (input) values in the
    training data.
    
    The distribution of visits, over possible next moves from each game state,
    made by the ZeroBot's select_move algorithm is also recorded. These become
    the Y (label) values in the training data for the policy head of the neural
    network.
    
    The player making the next move for each game-state and the winner of the
    game are also recorded and returned. These determine the reward to be used
    as a Y (label) value for the value head of the neural network.
    
    The game will start from the given starting position if one is provided.
    The game will end in a draw if the number of moves exceeds 'max_moves'.
    If 'max_moves' is zero, the game will continue until there is a winner.
    """
    
    if starting_board:
        game = GameState.new_game(starting_board)
    else:
        game = GameState.new_game()
        
    boards, moves, prior_targets, players = [], [], [], []
    if max_moves == 0:
        num_moves = -1
    else:
        max_moves = abs(max_moves)
        num_moves = 0
    
    while game.is_not_over() and num_moves < max_moves:
        bot = black if game.player == -1 else white
        
        # Get the bot to pick the next move and get the distribution of visits.
        if bot.is_trainable:
            action, tree_root = bot.select_move(game,
                                                return_search_tree=True)
            
            # Get the visit counts of the branches.
            visit_counts = {}
            for move in tree_root.branches.keys():
                visit_counts[move] = tree_root.branches[move].visit_count
            
            if np.random.rand() < eps:
                action = bot.rand_bot.select_move(game)
            
            if action.is_play:
                # Encode and record the game-state as well as the visit counts and
                # the player that made the move.
                # TODO: Record number of nodes (total visit counts) in the tree here also.
                board_tensor = bot.network.encoder.encode(game)
                boards.append(board_tensor)
                moves.append(action.move)
                prior_targets.append(visit_counts)
                players.append(game.player)
        else:
            action = bot.select_move(game)
            
        # Make the move. The select_move method should always return a legal
        # move.
        game.take_turn_with_no_checks(action)
        if max_moves > 0:
            num_moves += 1
        
    # if num_moves >= max_moves:
    #     print("Game ended in a draw.")
                
    return boards, moves, prior_targets, players, game.winner


def gain_experience(white, black, num_episodes, moves_limit = 0, eps = 0):
    """
    A function to repeatedly call the above simulate_game function in order to
    create a data set of games to train a ZeroBot on.
    
    The data from each game is stored in an 'episode' dictionary and all 
    episodes are collected into a list called 'experience' to be returned.
    """
    experience = []
    w = 0; b = 0
    
    for i in range(num_episodes):
        print('\rPlaying game {0}. Wins - w:{1} b:{2}'.format(i, w, b), end='')
        
        # Initialise the current episode as an empty dictionary
        episode = {}
            
        # Play a game and collect the generated data into the episode 
        # dictionary.
        game_details = simulate_game(white, black, None, moves_limit, eps)
        boards, moves, prior_targets, players, winner = game_details
        
        episode['boards'] = boards
        episode['moves'] = moves
        episode['prior_targets'] = prior_targets
        episode['players'] = players
        episode['winner'] = winner
        
        if winner == 1: 
            w += 1
        elif winner == -1:
            b += 1
        
        experience.append(episode)
    
    print('\rFinished playing. Wins - w:{0} b:{1}'.format(w, b))
    return experience


def competition(bot_a, bot_b, num_games=350, moves_limit=140, threshold=0):
    score = 0
    for g in range(num_games):
        
        if g%2 == 0:
            white, black = bot_a, bot_b
        else:
            white, black = bot_b, bot_a
            
        game_details = simulate_game(white, black, None, moves_limit, 0)
        boards, moves, prior_targets, players, winner = game_details
        
        if winner == 1:
            score += 1 if g%2 == 0 else -1
        elif winner == -1:
            score += 1 if g%2 == 1 else -1
            
        print('\rAfter {0} games the score is {1}.'.format(g+1, score), end='')
            
    print(' Competition finished.')
    if score > threshold:
        return bot_a
    elif score < threshold:
        return bot_b
    else:
        return None
        
        
        


def create_training_data(bot, experience):
    """
    A function to convert game data in an experience list to training data
    for training the ZeroBot neural network. The training data is also
    expanded 8 fold using symetries of the game. 
    """
    
    # The network input and labels forming the training set will be stored in
    # the following lists.
    X, Y, rewards = [], [], []
    
    # For each episode in the experience append the relevant tensors to the X,
    # Y and reward lsits.
    for episode in experience:
        
        Xi =  np.array(episode['boards']) 
        num_moves = Xi.shape[0]
        X.append(Xi)
        
        visit_counts = episode['prior_targets']
        policy_targets = bot.network.encoder.encode_priors(visit_counts)
        
        # The reward for moves decays exponentially with the number of moves
        # between it and the winning move. Rewards for moves made by the 
        # winning side are positive and negative for the losing side.
        episode_rewards = episode['winner'] * np.array(episode['players'])
        episode_rewards = (np.exp(-1*(num_moves-np.arange(num_moves)-1)/40
                                 )) * episode_rewards
        
        rewards.append( episode_rewards )
        
        Y.append( policy_targets )
      
    # Convert the X, Y lists into numpy arrays
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    
    # Use the bot's game encoder to expand the training data 8 fold.
    X, Y = bot.network.encoder.expand_data(X, Y)
    rewards = np.concatenate(8*rewards)
        
    return X, Y, rewards


def save_training_data(training_data, cycle):
    if not os.path.exists('data_bank/training_set_{0}'.format(cycle)):
        os.makedirs('data_bank/training_set_{0}'.format(cycle))
            
    for i, variable in enumerate(training_data):
        filename = 'data_bank/training_set_{0}/variable_{1}'.format(cycle, i)
        np.save(filename, variable)
        
def load_training_data():
    training_data = {}
    cycle = 0
    while os.path.exists('data_bank/training_set_{0}'.format(cycle)):
        i = 0
        variable_name = 'data_bank/training_set_{0}/variable_{1}.npy'.format(cycle, i)
        while os.path.exists(variable_name):
            if i in training_data.keys():
                training_data[i] = np.concatenate([training_data[i], np.load(variable_name)])
            else:
                training_data[i] = np.load(variable_name)
            i += 1
            variable_name = 'data_bank/training_set_{0}/variable_{1}.npy'.format(cycle, i)
        cycle += 1
        
    training_data = tuple(training_data[j] for j in range(i))
    return training_data
    


def random_starting_position(num_white_pieces, num_black_pieces):
    """
    Function to randomly select a starting position for a game a number of
    white and black pieces given by 'num_white_pieces' and 'num_black_pieces'
    respectively.
    """
    board = {}
    for i in range(7):
        for j in range(7):
            board[(i,j)] = 0

    # Set up the black pieces
    black_positions = [(3, 0), (3, 1), (3, 5), (3, 6),
                       (0, 3), (1, 3), (5, 3), (6, 3)]
    for square in random.sample(black_positions, num_black_pieces):
        board[square] = -1

    # Set up the white pieces
    white_positions = [(3, 2), (3, 3), (2, 3), (4, 3)]
    for square in random.sample(white_positions, num_white_pieces):
        board[square] = 1

    # Place the king piece in the centre
    board[(3,3)] = 2
    
    return board


class DataManager():
    
    def __init__(self):
        self.Xs = None
        self.Ys = None
        self.Rs = None
        self.appended = []
        self.max_bank_size = 140000
        
    def append_data(self, training_data):
        X, Y, R = training_data
        if type(self.Xs) is np.ndarray:
            self.Xs = np.concatenate((self.Xs, X))
            self.Ys = np.concatenate((self.Ys, Y))
            self.Rs = np.concatenate((self.Rs, R))
        else:
            self.Xs = X
            self.Ys = Y
            self.Rs = R
        
        self.appended.append(len(X))
            
        if self.Xs.shape[0] > self.max_bank_size:
            self.Xs = self.Xs[-self.max_bank_size:, :, :, :]
            self.Ys = self.Ys[-self.max_bank_size:, :, :, :]
            self.Rs = self.Rs[-self.max_bank_size:]
            
    def sample_training_data(self, num=4096):
        samples = np.random.choice(len(self.Xs), num)
        return self.Xs[samples, :, :, :], self.Ys[samples, :, :, :], self.Rs[samples]