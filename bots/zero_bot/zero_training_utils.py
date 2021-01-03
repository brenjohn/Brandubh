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

from brandubh import GameState



def simulate_game(bot, starting_board=None, max_moves=0):
    """
    A function to get the provided bot to play a single game of brandubh
    against itself in order to generate training data for the bot.
    
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
    
    TODO: moves no longer need to be recorded to create training data.
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
        # Get the bot to pick the next move and get the distribution of visits.
        action, visit_counts = bot.select_move(game, 
                                               return_visit_counts=True)
        
        if action.is_play:
            # Encode and record the game-state as well as the visit counts and
            # the player that made the move.
            board_tensor = bot.network.encoder.encode(game)
            boards.append(board_tensor)
            moves.append(action.move)
            prior_targets.append(visit_counts)
            players.append(game.player)
            
        # Make the move. The select_move method should always return a legal
        # move.
        game.take_turn_with_no_checks(action)
        if max_moves > 0:
            num_moves += 1
        
    if num_moves >= max_moves:
        print("Game ended in a draw.")
                
    return boards, moves, prior_targets, players, game.winner


def gain_experience(bot, num_episodes, max_num_white_pieces = None, 
                                       max_num_black_pieces = None,
                                       moves_limit = None):
    """
    A function to repeatedly call the above simulate_game function in order to
    create a data set of games to train a ZeroBot on.
    
    The data from each game is stored in an 'episode' dictionary and all 
    episodes are collected into a list called 'experience' to be returned.
    """
    
    experience = []
    
    for i in range(num_episodes):
        print('\rPlaying game {0}. '.format(i),end='')
        
        # Initialise the current episode as an empty dictionary
        episode = {}
        
        # If a maximum number of white and black pieces is provided,
        # a random starting position for the episode/game is selected, 
        # otherwise the standard starting position is used.
        if max_num_black_pieces == None or max_num_white_pieces == None:
            board = None
        else:
            num_white_pieces = random.randint(0, max_num_white_pieces)
            num_black_pieces = random.randint(1, max_num_black_pieces)
            message = "The game began with {0} white pieces and {1} black. "
            print(message.format(num_white_pieces, num_black_pieces),end='')
            board = random_starting_position(num_white_pieces,
                                             num_black_pieces)
            
        # Play a game and collect the generated data into the episode 
        # dictionary.
        game_details = simulate_game(bot, board, moves_limit)
        boards, moves, prior_targets, players, winner = game_details
        
        episode['boards'] = boards
        episode['moves'] = moves
        episode['prior_targets'] = prior_targets
        episode['players'] = players
        episode['winner'] = winner
        
        experience.append(episode)
    
    print('. Playing finished')
    return experience


def create_training_data(bot, experience):
    """
    A function to convert game data in an experience list to training data
    for training the ZeroBot neural network. The training data is also
    expanded 8 fold using sysmetries of the game. 
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