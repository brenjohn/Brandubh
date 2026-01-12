#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 14:25:39 2025

@author: john

This module define a Trainer class for running and managing the ZeroBot
training loop.
"""

import json

from ...game import GameState
from ..random_bot import RandomBot


class Trainer:
    """The Trainer class is responsible for orchestrating the training loop
    for the brandubh zero bot. 
    
    A training iteration takes the following steps:
        1 - The bot is used to generate several self play games.
        2 - The self play games are converted to training data and appened to a
            buffer managed by a data_manager object.
        3 - A sample of training data is collected from the buffer and used as
            a dataset to train the bot for an epoch.
        4 - The bot is evaluated against selected opponent bots.
    """
    
    def __init__(
            self,
            output_dir,
            zero_bot,
            data_manager,
            evaluator,
            random_policy,
            num_cycles,
            episodes_per_cycle,
            move_limit,
            batch_size,
            **kwargs
        ):
        self.output_dir         = output_dir
        self.bot                = zero_bot
        self.data_manager       = data_manager
        self.evaluator          = evaluator
        self.random_policy      = random_policy
        self.num_cycles         = num_cycles
        self.episodes_per_cycle = episodes_per_cycle
        self.move_limit         = move_limit
        self.batch_size         = batch_size
        
        self.balance_history = { 
            'white_win_moves' : [], 
            'black_win_moves' : [], 
            'draw_moves'  : []
        }
        self.loss_history = {}
        self.log_data = {
            'loss_history'    : self.loss_history,
            'balance_history' : self.balance_history
        }
        self.curr_model_dir = output_dir / 'model/'
        self.curr_model_dir.mkdir(exist_ok=True)
        self.experience_dir = output_dir / 'experience/'
        self.experience_dir.mkdir(exist_ok=True)
    
    
    def train(self):
        """Runs the zero bot training loop.
        """
        bot = self.bot
        encoder = bot.get_encoder()
        data_manager = self.data_manager
        evaluator = self.evaluator
        
        cycle = -1
        while True:
            cycle += 1
            
            # Collect brandubh experience through self play games.
            print('\nGainning experience, cycle {0}'.format(cycle))
            experience, balance = self.gain_experience()
            save_experience(self.experience_dir, cycle, experience)
            
            # Convert collected experience into training data.
            print('Preparing training data')
            training_data = encoder.create_training_data(experience)
            data_manager.append(training_data)
            
            # Sample some of the collected training data and train on it.
            print('\nTraining network, cycle {0}'.format(cycle))
            training_data = data_manager.sample_training_data()
            loss = bot.network.train(training_data, batch_size=self.batch_size)
            
            # Save current bot and log the loss and balance measurements.
            update_log_dict(self.balance_history, balance)
            update_log_dict(self.loss_history, loss)
            bot.save_bot(self.curr_model_dir)
            self.save_training_log()
            
            # Evaluate the current bot.
            if evaluator.should_evaluate(cycle):
                print('\nEvaluating bot, cycle {0}'.format(cycle))
                evaluator.evaluate(bot)
            
            # Update random move policy.
            self.random_policy.update(self.log_data)
            
            # Cheak if the training loop should be broken.
            if cycle == self.num_cycles:
                break
                
                
    def save_training_log(self):
        log_to_save = {**self.log_data}
        log_to_save['random_policy'] = self.random_policy.get_state_dict()
        with open(self.output_dir / 'training_log.json', 'w') as file:
            json.dump(log_to_save, file, indent=2)
    
    
    def gain_experience(self):
        """Generates a list of self play games for the bot to train on.
        """
        experience = []
        white_wins = 0; black_wins = 0
        balance = { 
            'white_win_moves' : 0, 
            'black_win_moves' : 0, 
            'draw_moves'  : 0
        }
        
        message = '\rPlaying game {0}. Wins - w:{1} b:{2}'
        for i in range(self.episodes_per_cycle):
            print(message.format(i, white_wins, black_wins), end='')
                
            # Play a game and collect the generated data.
            episode = self_play(self.bot, self.random_policy, self.move_limit)
            experience.append(episode)
            
            if episode['winner'] == 1: 
                white_wins += 1
                balance['white_win_moves'] += episode['players'].count(1)
                
            elif episode['winner'] == -1:
                black_wins += 1
                balance['black_win_moves'] += episode['players'].count(-1)
                
            else:
                balance['draw_moves'] += len(episode['players'])
        
        message = '\rFinished playing {0} games. Wins - w:{1} b:{2}'
        print(message.format(i+1, white_wins, black_wins))
        return experience, balance


def self_play(bot, random_policy, max_moves=0):
    """Gets the provided bot to play a single game of brandubh against itself 
    and returns a dict containing the game. To improve variation of game
    states, random moves are occasionally selected according to the given
    random move policy (See RandomMovePolicy classes).
    
    The retunred dict contains the following ordered lists:
        
        boards - A list of encoded game states that occured during the game. 
        These can be used as input (X) values in training data for the ZeroBot.
        
        moves_played - A list of all moves played during the game.
        
        visit_counts - A list of distributions of ZeroBot tree search visits
        over legal moves each turn. These can be used as target (Y) values in
        training data for the policy head of the ZeroBot neural network.
        
        tree_stats - A list of dicts containing various tree search statistics
        for each turn. These con be used to inspect predictions made by the 
        ZeroBot
        
        players - A list of ints indicating the player making a move.
        
        winner - An int indicating who won the game. This can be used to create
        target values (rewards) for in trainging data for the value head of the
        ZeroBot neural network.
    
    The game will end in a draw if the number of moves exceeds 'max_moves'.
    """
    rand_bot = RandomBot()
    encoder = bot.get_encoder()
    game = GameState.new_game()
        
    boards, moves_played, move_priors, tree_stats, players = [], [], [], [], []
    random_move = []
    num_moves = 0
    
    while game.is_not_over() and num_moves < max_moves:
        
        # Get the bot to pick the next move and get the distribution of visits.
        action = bot.select_move(game)
        
        # Get the visit counts of the branches.
        tree_root = bot.root
        moves, _ = game.legal_moves()
        stats = tree_root.get_search_stats(moves)
        visit_counts = tree_root.visit_counts()
        
        random_move.append(False)
        if random_policy.make_random_move(game):
            action = rand_bot.select_move(game)
            random_move[-1] = True
        
        if action.is_play:
            # Encode and record the game-state as well as the visit counts and
            # the player that made the move.
            board_tensor = encoder.encode(game)
            boards.append(board_tensor)
            moves_played.append(action.move)
            tree_stats.append(stats)
            move_priors.append(visit_counts)
            players.append(game.player)
            
        # Make the move. The select_move method should always return a legal
        # move.
        game.take_turn_with_no_checks(action)
        num_moves += 1
                
    return {
        'boards'       : boards, 
        'moves_played' : moves_played, 
        'visit_counts' : move_priors, 
        'tree_stats'   : tree_stats, 
        'players'      : players, 
        'winner'       : game.winner,
        'random_move'  : random_move
    }



def save_experience(output_dir, cycle, experience):
    """Save the given self play experience in json format.
    """
    output_dir = output_dir / f'cycle_{cycle}_experience/'
    output_dir.mkdir(exist_ok=True)
    
    for num, episode in enumerate(experience):
        episode = episode.copy()
        episode['boards'] = [board.tolist() for board in episode['boards']]
        del episode['visit_counts']
        
        filename = output_dir / f'game_{num}.json'
        with open(filename, 'w') as file:
            json.dump(episode, file, indent=2)



def update_log_dict(history, loss):
    for key, value in loss.items():
        if isinstance(value, dict):
            update_log_dict(history.setdefault(key, {}), value)
        elif isinstance(value, list):
            # Here we assume a training cycle trains for exactly 1 epoch, and 
            # so there's only one loss value per cycle. If this is ever changes 
            # and we want to record the loss value for several epochs every 
            # cycle, we can use append instead.
            history.setdefault(key, []).extend(value)
        elif isinstance(value, int): 
            history.setdefault(key, []).append(value)