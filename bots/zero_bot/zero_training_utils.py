#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:28:27 2020

@author: john
"""
import numpy as np
import random

from brandubh import GameState



def simulate_game(bot, starting_board=None):
    
    if starting_board:
        game = GameState.new_game(starting_board)
    else:
        game = GameState.new_game()
        
    boards, moves, prior_targets, players = [], [], [], []
    
    while game.is_not_over():
        action, visit_counts = bot.select_move(game, return_visit_counts=True)
        
        if action.is_play:
            board_tensor = bot.encoder.encode(game)
            boards.append(board_tensor)
        
            # Can probably avoid calc pieces here by getting return move to
            # return move index
            pieces = np.sum(board_tensor[:,:,:2],-1).reshape((7,7))
            moves.append(bot.encoder.encode_move(pieces, action.move))
            
            prior_targets.append(visit_counts)
            
            players.append(game.player)
            
        # Make the move.
        game.take_turn_with_no_checks(action)
                
    return boards, moves, prior_targets, players, game.winner


def gain_experience(bot, num_episodes, num_white_pieces = None, 
                                       num_black_pieces = None):
    
    experience = []
    
    for i in range(num_episodes):
        print('\rPlaying game {0}'.format(i),end='')
        episode = {'boards': [],
                  'moves': [],
                  'prior_targets': [],
                  'players': [],
                  'winner': 0}
        
        if num_black_pieces == None or num_white_pieces == None:
            board = None
        else:
            board = random_starting_position(num_white_pieces,
                                             num_black_pieces)
            
        boards, moves, prior_targets, players, winner = simulate_game(bot, 
                                                                      board)
        
        episode['boards'] = boards
        episode['moves'] = moves
        episode['prior_targets'] = prior_targets
        episode['players'] = players
        episode['winner'] = winner
        
        experience.append(episode)
    
    print('. Playing finished')
    return experience


def create_training_data(bot, experience):
    
    X, Y, rewards = [], [], []
    
    for episode in experience:
        
        Xi =  np.array(episode['boards']) 
        num_moves = Xi.shape[0]
        X.append(Xi)
        
        visit_counts = episode['prior_targets']
        total_visits = np.sum(visit_counts, axis=1).reshape(num_moves, 1)
        policy_targets = visit_counts / total_visits
        
        rewards.append( episode['winner'] * np.array(episode['players']) )
        
        Y.append( policy_targets )
        
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    
    X, Y = bot.encoder.expand_data(X, Y)
    rewards = np.concatenate(8*rewards)
        
    return X, Y, rewards


def random_starting_position(num_white_pieces, num_black_pieces):
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