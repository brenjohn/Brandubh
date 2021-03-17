#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:11:37 2020

@author: john

Running this script provides the user with a terminal interface for setting
up and playing a game of brandubh. 
"""

import time
import utils
import os

from brandubh import GameState
from bots.random_bot import RandomBot
from bots.mcbot import MCTSBot
from bots.zero_bot.brandubh_zero import ZeroBot
from bots.zero_bot.zero_network import ZeroNet

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_a_zero_bot():
    """
    A function for creating/loading a ZeroBot to act as a player in a game of
    brandubh.
    """
    bot = ZeroBot()
    if os.path.exists("bots/zero_bot/model_data/"):
        bot.load_bot("bots/zero_bot/model_data/")
    else:
        bot.network = ZeroNet()
            
    return bot

# A dictionary used to map input from the user to constructors for different
# agents that may play in a game.
PLAYERS = {"user": utils.Player,
           "rand": RandomBot,
           "mcbot": MCTSBot,
           "zero": create_a_zero_bot}



def start_a_new_game(white, black):
    """
    This function creates a new game of brandubh and asks the players
    to select moves until the game is over. The arguments 'white' and 'black'
    should be agents that implement a select_move method which returns a move
    to be made next, given an arbitrary game-state/board-position.
    """
    game = GameState.new_game()
    next_move = ' '
    
    while game.is_not_over():
        # Clear the output screen and print the current board position
        print(chr(27) + "[2J")
        utils.print_board(game, next_move)
        time.sleep(0.5)
            
        # Ask the proper agent what the next move should be.
        if game.player == 1:
            action = white.select_move(game)
            move = action.move
        else:
            action = black.select_move(game)
            move = action.move
         
        # Create a string indicating what the selected move was to be 
        # displayed on screen for the next turn.
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
    This function provides the user with an interface for creating and
    interacting with a game of brandubh.
    """
    
    while True:
        # Print the main menu and wait for input from the user.
        print(chr(27) + "[2J")
        print(utils.TITLE)
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
        
        # Run the appropriate commands for the option selected by the user.
        if option == 'exit':
            break
        
        elif option == 'rules':
            print(chr(27) + "[2J")
            print(utils.RULEBOOK)
            time.sleep(0.2)
            input('--')
            
        elif option == 'about':
            print(chr(27) + "[2J")
            print('Put game history here')
            time.sleep(0.2)
            input('--')
            
        elif option == 'play':
            print(chr(27) + "[2J")
            
            # Ask the user which agents should play for the black and white 
            # sides.
            black_player = ''
            while black_player not in PLAYERS:
                time.sleep(0.2)
                black_player = input('Who will play as black?' + \
                                     ' (user, rand, mcbot, or zero) -')
                
            white_player = ''
            while white_player not in PLAYERS:
                time.sleep(0.2)
                white_player = input('Who will play as white?' + \
                                     ' (user, rand, mcbot or zero) -')
                    
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