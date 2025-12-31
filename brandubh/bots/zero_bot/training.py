#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 14:25:39 2025

@author: john
"""

import json
import numpy as np

from ...game import GameState
from ..random_bot import RandomBot


def self_play(bot, starting_board=None, max_moves=0, eps=0):
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
    """
    rand_bot = RandomBot()
    game = GameState.new_game(starting_board)
        
    boards, moves, prior_targets, players = [], [], [], []
    num_moves = 0
    
    while game.is_not_over() and num_moves < max_moves:
        
        # Get the bot to pick the next move and get the distribution of visits.
        action = bot.select_move(game)
        tree_root = bot.root
        
        # Get the visit counts of the branches.
        visit_counts = {}
        for move in tree_root.branches.keys():
            visit_counts[move] = tree_root.branches[move].visit_count
        
        if np.random.rand() < eps:
            action = rand_bot.select_move(game)
        
        if action.is_play:
            # Encode and record the game-state as well as the visit counts and
            # the player that made the move.
            # TODO: Record number of nodes (total visit counts) in the tree here also.
            board_tensor = bot.network.encoder.encode(game)
            boards.append(board_tensor)
            moves.append(action.move)
            prior_targets.append(visit_counts)
            players.append(game.player)
            
        # Make the move. The select_move method should always return a legal
        # move.
        game.take_turn_with_no_checks(action)
        num_moves += 1
                
    return boards, moves, prior_targets, players, game.winner



def gain_experience(bot, num_episodes, moves_limit = 0, eps = 0):
    """
    A function to repeatedly call the above simulate_game function in order to
    create a data set of games to train a ZeroBot on.
    
    The data from each game is stored in an 'episode' dictionary and all 
    episodes are collected into a list called 'experience' to be returned.
    """
    experience = []
    white_wins = 0; black_wins = 0
    
    message = '\rPlaying game {0}. Wins - w:{1} b:{2}'
    for i in range(num_episodes):
        print(message.format(i, white_wins, black_wins), end='')
            
        # Play a game and collect the generated data.
        game_details = self_play(bot, None, moves_limit, eps)
        boards, moves_played, visit_counts, players, winner = game_details
        
        episode = {}
        episode['boards']       = boards
        episode['moves_played'] = moves_played
        episode['visit_counts'] = visit_counts
        episode['players']      = players
        episode['winner']       = winner
        experience.append(episode)
        
        if winner == 1: 
            white_wins += 1
        elif winner == -1:
            black_wins += 1
    
    message = '\rFinished playing {0} games. Wins - w:{1} b:{2}'
    print(message.format(num_episodes, white_wins, black_wins))
    return experience



def save_experience(output_dir, cycle, experience):
    output_dir = output_dir / f'cycle_{cycle}_experience/'
    output_dir.mkdir(exist_ok=True)
    
    for num, episode in enumerate(experience):
        episode = episode.copy()
        episode['boards'] = [board.tolist() for board in episode['boards']]
        episode['visit_counts'] = [{
            str(move) : count 
            for move, count in visit_counts.items()
        } for visit_counts in episode['visit_counts']]
        
        filename = output_dir / f'game_{num}.json'
        with open(filename, 'w') as file:
            json.dump(episode, file, indent=2)