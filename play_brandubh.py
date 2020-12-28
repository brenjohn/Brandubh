#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:11:37 2020

@author: john
"""

import time
import utils

from brandubh import GameState
from bots.random_bot import RandomBot
from bots.zero_bot.brandubh_zero import ZeroBot



def create_a_zero_bot():
    bot = ZeroBot()
    bot.load_bot("bots/zero_bot/model_data/")
    return bot

PLAYERS = {"user": utils.Player,
           "rand": RandomBot,
           "zero": create_a_zero_bot}



def start_a_new_game(white, black):
    """
    This function creates a new game of brandubh and asks the players
    to select moves until the game is over.
    """
    
    game = GameState.new_game()
    next_move = ' '
    
    while game.is_not_over():
        
        # Clear the output screen and print the current board position
        print(chr(27) + "[2J")
        utils.print_board(game, next_move)
        time.sleep(0.5)
            
        if game.player == 1:
            action = white.select_move(game)
            move = action.move
        else:
            action = black.select_move(game)
            move = action.move
            
        if move:
            next_move = utils.COLS[move[1]] + str(move[0]) + ' ' + \
                        utils.COLS[move[3]] + str(move[2])
        else:
            next_move = 'pass'
         
        # Try make the move and see if it is legal. If not, print
        # why it isn't to the screen and wait the next input
        # (Note, all actions produced by select_move methods should be legal,
        # this is just here to catch errors.)
        move_is_not_legal = game.take_turn(action)
        if move_is_not_legal:
            print(move_is_not_legal)
       
    # If the game is over, print the winning board position and who won             
    print(chr(27) + "[2J")
    utils.print_board(game, next_move)
    winner = 'black' if game.winner==-1 else 'white'
    print('The winner is ' + winner)
    time.sleep(0.2)
    input('--')



def main():
    """
    This function provides a user interface for brandubh.
    """
    
    while True:
        print(chr(27) + "[2J")
        print(utils.title)
        print('F\u00e1ilte go bradubh / Welcome to brandubh')
        print('.......................................')
        print(' ')
        print('Enter one of the following options:')
        print('play  - play a game of brandubh')
        print('rules - See the rules of the game')
        print('about - Some history behind brandubh')
        print('exit  - quit the game')
        time.sleep(0.2)
        option = input('-- ')
        
        if option == 'exit':
            break
        
        elif option == 'rules':
            print(chr(27) + "[2J")
            print('Put game rules here')
            time.sleep(0.2)
            input('--')
            
        elif option == 'about':
            print(chr(27) + "[2J")
            print('Put game history here')
            time.sleep(0.2)
            input('--')
            
        elif option == 'play':
            print(chr(27) + "[2J")
            
            # Ask the user what pieces they want to use
            black_player = ''
            while black_player not in PLAYERS:
                time.sleep(0.2)
                black_player = input('Who will play as black?' + \
                                     ' (user, rand or zero) -')
                
            white_player = ''
            while white_player not in PLAYERS:
                time.sleep(0.2)
                white_player = input('Who will play as white?' + \
                                     ' (user, rand or zero) -')
                    
            black_player = PLAYERS[black_player]()
            white_player = PLAYERS[white_player]()
                
            start_a_new_game(white_player, black_player)
            
        else:
            print(chr(27) + "[2J")
            print('Invalid option')
            time.sleep(0.2)
            input('--')
            
    print(chr(27) + "[2J")
    print('Sl\u00e1n / Goodbye')

        
        
if __name__ == '__main__':
    main()