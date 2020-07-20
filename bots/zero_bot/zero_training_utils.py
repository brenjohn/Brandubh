#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:28:27 2020

@author: john
"""
import numpy as np

from brandubh import GameState



def simulate_game(bot):
    
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


def gain_experience(bot, num_episodes):
    
    experience = []
    
    for i in range(num_episodes):
        print('\rPlaying game {0}'.format(i),end='')
        episode = {'boards': [],
                  'moves': [],
                  'prior_targets': [],
                  'players': [],
                  'winner': 0}
        
        boards, moves, prior_targets, players, winner = simulate_game(bot)
        
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