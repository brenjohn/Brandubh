#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:10:56 2021

@author: john

This script creates a UI to allow the user to create and start a game of
brandubh. It is responsibile for handling user input, passing instructions to
both the model (ie the GameState of the game and any bots playing), to progress
the game, and the view, to render information on the terminal screen.

This script uses a curses standard screen (stdscr) to get user input and calls
functions imported from 'brandubh_view.py' to render information on it.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import curses

from brandubh import GameState, Act

from UI_utils.brandubh_view import init_view, draw_main_menu, draw_rulebook
from UI_utils.brandubh_view import draw_player_selection_screen
from UI_utils.brandubh_view import draw_loading_screen
from UI_utils.brandubh_view import draw_game_screen, draw_game_over_banner

from bots.random_bot import RandomBot
from bots.mcbot import MCTSBot
from bots.greedy_random_bot import GreedyRandomBot
from bots.zero_bot.brandubh_zero import ZeroBot


# The bots that can be selected to play a game of brandubh.
PLAYERS = ["user",
           RandomBot,
           GreedyRandomBot,
           MCTSBot,
           ZeroBot]



def main_menu(stdscr):
    """
    This function controls the main menu screen where the user can select to
    play a game, read the rules of the game or exit the game. It also
    initialises curses variables that are used for the duration of the game.
    """
    # Initialise the screen colours and cursor, the key variable recording the 
    # last key entered by the user and the current option from the main menu.
    init_view(stdscr)
    key = 0
    option = 0

    # Infinite loop to repeatedly ask the user to enter a new key.
    while True:
        # If the user hit the 'enter' key, select the choosen option,
        if key == 10:
            if option == 0:
                start_game(stdscr)
                key = 0
                continue
            
            # If the chosen option was to read the rules, open rulebook.
            if option == 1:
                view_rulebook(stdscr)
                key = 0
                continue
            
            # If the chosen option was to quit, then exit the function.
            if option == 2:
                return
            
        # Else if the user hit the up or down arrow keys, increment the 
        # the current option.
        elif key == curses.KEY_DOWN:
            option = (option + 1)%3
        elif key == curses.KEY_UP:
            option = (option - 1)%3
            
        # Draw the main menu with the current option highlighted.
        draw_main_menu(stdscr, option)

        # Wait for next input from the user.
        key = stdscr.getch()
        


def start_game(stdscr):
    """
    Asks the user to select players for the white and black side and then
    starts a new game of brandubh with the selected players.
    """
    black_player, white_player = select_players(stdscr)
    play_game(stdscr, black_player, white_player)
    
  
    
def select_players(stdscr):
    """
    This function controls the screen where the user can select which players
    will play as white and black. The user can select to play as either/both
    white or/and black or select a bot to play as either side.
    """
    # Initialise the key variable recording the last key entered by the user 
    # and the current option from the menu.
    key = 0
    option = 0
    players = []

    # Infinite loop to repeatedly ask for the user to enter a new key.
    while True:
        
        # If the user hit the 'enter' key, select the choosen option,
        if key == 10:
            # When 2 players have been selected, move on.
            if len(players) == 2:
                draw_loading_screen(stdscr)
                break
            else:
                players.append(option)
                
        # If the user pressed 'backspace' then undo the last selection.
        elif key == 127:
            if len(players) > 0:
                players.pop()
            
        # Else if the user with the up or down arrow keys, increment the 
        # the current option.
        elif key == curses.KEY_DOWN:
            option = (option + 1) % len(PLAYERS)
        elif key == curses.KEY_UP:
            option = (option - 1) % len(PLAYERS)
        
        # Draw the main menu with the current option highlighted.
        draw_player_selection_screen(stdscr, players, option)

        # Wait for next input from the user.
        key = stdscr.getch()
        
    # Return instances of the selected bots or the string "user" if the user
    # wants to play.
    black_player, white_player = PLAYERS[players[0]], PLAYERS[players[1]]
    if black_player != "user":
        black_player = black_player()
    if white_player != "user":
        white_player = white_player()
    return black_player, white_player
        
    
        
def play_game(stdscr, black="user", white="user"):
    """
    This function starts and controls a game of brandubh. It continuely asks
    the provided players to select the next move in the game and invokes the
    appropriate functions to update the game state and render the current 
    board on the screen.
    """
    # Initialise variables to track the last key pressed, the cursor location
    # on the game board, an array of coordinates specifying the next move and
    # create an new game of brandubh.
    key = 0
    cursor_x = 0
    cursor_y = 0
    next_move = []
    bot_to_take_next_turn = False
    game = GameState.new_game()

    # Loop to continuely ask the players to select the next move and update
    # the game state until 'q' is pressed to quit the game.
    while (key != ord('q')):
        # Ask the proper agent to select the next move.
        agent = white if game.player == 1 else black
        if agent == "user":
            tmp = update_user_variables(key, cursor_x, cursor_y, next_move)
            action, cursor_x, cursor_y, next_move = tmp
        else:
            if bot_to_take_next_turn:
                action = agent.select_move(game)
                bot_to_take_next_turn = False
            else:
                bot_to_take_next_turn = True
                action = None
        
        # Try make the move and see if it is legal. If not, print
        # why it isn't to the screen and wait the next move to be selected.
        if action:
            info_message = game.take_turn(action)
        elif bot_to_take_next_turn:
            side = "White" if game.player == 1 else "Black"
            info_message = "{0} is thinking.".format(side)
        else:
            info_message = None
        
        # Get the last 6 moves from the game history to print to the screen.
        last_6_moves = []
        old_state = game.history
        while old_state.last_move != None and len(last_6_moves) < 6:
            last_6_moves.append((old_state.turn, old_state.last_move))
            old_state = old_state.previous_state
        
        # Draw the current state of the board on the screen along with any
        # other relevant information for the user.
        board_pieces = game.game_set.board
        draw_game_screen(stdscr, board_pieces, 
                         cursor_x, cursor_y, 
                         game.player, next_move, last_6_moves,
                         info_message)
        
        # If the game is over then print a game over message to the screen.
        if not game.is_not_over():
            draw_game_over_banner(stdscr, game.winner)

        # Wait for next user input if a bot is not about to be asked to select
        # a move.
        if not bot_to_take_next_turn:
            key = stdscr.getch()
    
    
        
def update_user_variables(key, cursor_x, cursor_y, next_move):
    """
    This function is used to ask the user to select the next move in a game
    and returns the currect action if one is selected. It also updates 
    variables used to track which move the user selects. Namely, it updates
    the cursor location when an arrow key is pressed and appends/pops the 
    current location of the cursor to/from the 'next_move' array when 
    'Enter'/'backspace' is pressed.
    """
    action = None
    # If enter key was hit, update move variables
    if key == 10:
        if len(next_move) < 2:
            next_move.append((cursor_x, cursor_y))
        else:
            action = Act.play(next_move[0] + next_move[1])
            next_move = []
            
    # If 'backspace' was pressed, undo previous updates to the move variables.
    elif key == 127:
        if len(next_move) > 0:
            next_move.pop()
            
    # If 'p' was pressed, pass the turn.
    elif key == ord('p'):
        action = Act.pass_turn()
        
    # If 'r' was pressed, resign from the game.
    elif key == ord('r'):
        action = Act.resign()
        
    # If an arrow key was pressed, update the cursor location.
    elif key == curses.KEY_DOWN:
        cursor_y = cursor_y + 1
    elif key == curses.KEY_UP:
        cursor_y = cursor_y - 1
    elif key == curses.KEY_RIGHT:
        cursor_x = cursor_x + 1
    elif key == curses.KEY_LEFT:
        cursor_x = cursor_x - 1
    cursor_x = cursor_x%7 # The brandubh board is a 7x7 board.
    cursor_y = cursor_y%7
    
    return action, cursor_x, cursor_y, next_move



def view_rulebook(stdscr):
    key = 0
    page = 0
    
    # Loop to continuely wait for input from the user until  'q' is pressed.
    while (key != ord('q')):
        # Else if the user hit the up or down arrow keys, increment the 
        # the current option.
        if key == curses.KEY_RIGHT:
            page = min(8, page + 1)
        elif key == curses.KEY_LEFT:
            page = max(0, page - 1)
            
        # Draw the main menu with the current option highlighted.
        draw_rulebook(stdscr, page)
        
        # Wait for next input from the user.
        key = stdscr.getch()
        


def main():
    curses.wrapper(main_menu)
    print('Sl\u00e1n / Goodbye')

if __name__ == "__main__":
    main()