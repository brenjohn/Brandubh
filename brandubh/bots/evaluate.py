#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 16:07:32 2025

@author: john
"""

import json

from ..game import GameState
from .random_bot import RandomBot
from .greedy_random_bot import GreedyRandomBot
from .mcbot import MCTSBot


BOTS = {
    'RandomBot' : RandomBot,
    'GreedyRandomBot' : GreedyRandomBot,
    'MCTSBot' : MCTSBot
}


class Evaluator:
    """The evaluator class can be used to setup and store several opponent bots
    to evaluate a brandubh bot against. It also manages recording results from
    evaluations.
    """
    
    def __init__(self, output_dir, opponents):
        self.output_dir = output_dir
        self.opponents = opponents
        
        self.setup_bots()
        
        
    def setup_bots(self):
        """Setup the opponent bots to be used to evaluate a bot.
        """
        self.bots = {}
        self.output_files = {}
        for name, params in self.opponents.items():
            self.output_files[name] = self.output_dir / (name + '_eval.json')
            self.bots[name] = BOTS[params['type']](**params)
            params['score'] = []
            params['games_won_as_white'] = []
            params['games_won_as_black'] = []
            
            
    def evaluate(self, bot):
        """Evaluate the given bot against the stored opponent bots and save the
        results.
        """
        for name, params in self.opponents.items():
            print('Evaluation:', name)
            opponent_bot = self.bots[name]
            bot.turn_on_eval_mode(**params)
            results = evaluate_bot(bot, opponent_bot, **params)
            bot.turn_off_eval_mode()
            
            score, white_wins, black_wins = results
            params['score'].append(score)
            params['games_won_as_white'].append(white_wins)
            params['games_won_as_black'].append(black_wins)
            
            with open(self.output_files[name], 'w') as file:
                json.dump(params, file, indent=4)
            


def evaluate_bot(
        bot, 
        opponent_bot,
        num_games,
        turn_limit = 700,
        **kwargs
    ):
    """
    Evaluates the given bot against the given opponent bot by letting them play 
    a number of games against each other. 
    
    The number of games played is specified by 'num_games'. If the number of 
    turns taken in a game exceeds the given maximum, then the game ends and 
    drawn up as a win for the opponent bot.
    """
    bot_player = 1
    wins_as_black = 0
    wins_as_white = 0
    
    # Play 'num_games' games of brandubh
    for game_num in range(num_games):
        message = '\rPlaying game {0}, score: w = {1}, b = {2}.'
        print(message.format(game_num, wins_as_white, wins_as_black), end='')
        game = GameState.new_game()
        
        # Get both bots to play a game of brandubh.
        turns_taken = 0
        while game.is_not_over() and turns_taken < turn_limit:
            if game.player == bot_player:
                action = bot.select_move(game)
            else:
                action = opponent_bot.select_move(game)  
            game.take_turn_with_no_checks(action)
            turns_taken += 1
         
            
        # Keeping track of how many games the bot won as white and black.
        if turns_taken < turn_limit and bot_player == game.winner:
            if bot_player == 1:
                wins_as_white += 1
            else:
                wins_as_black += 1
        
        # The bots should switch sides for the next game.
        bot_player *= -1
    
    message = '\rFinished playing {0} games. Score: w = {1}, b = {2}.'
    print(message.format(num_games, wins_as_white, wins_as_black))
    
    # Return the bot's average score along with the number of times it won as 
    # white and black.
    score = 2 * (wins_as_white + wins_as_black) / num_games - 1
    return score, wins_as_white, wins_as_black