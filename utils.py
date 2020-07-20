#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 09:08:53 2020

@author: john
"""

import re

from brandubh import Act

# This string and dictionary are used for printing an image of the board
COLS = 'ABCDEFG'
PIECE_TO_CHAR = {
    0. : '.',
    1. : 'o',
    -1.: 'x',
    2. : 'A'}

title = """
 ___                           _         _      _      
(  _`\                        ( )       ( )    ( )     
| (_) ) _ __   _ _   ___     _| | _   _ | |_   | |__   
|  _ <'( '__)/'_` )/' _ `\ /'_` |( ) ( )| '_`\ |  _ `\ 
| (_) )| |  ( (_| || ( ) |( (_| || (_) || |_) )| | | | 
(____/'(_)  `\__,_)(_) (_)`\__,_)`\___/'(_,__/'(_) (_) 
"""


def print_board(game, move):
    """
    This function will print an image of the given board position
    """
    game_set = game.game_set
    
    # Create strings indicating who's move it is and who moved last
    if game.player == -1:
        player = 'black'
        previous_move = 'white'
    else:
        player = 'white'
        previous_move = 'black'
    
    # Print the board
    print('   ' + ' '.join(COLS) + '        It is ' + player + "'s turn")
    print(' +---------------+      previous move: '+previous_move+' '+move)
    for row in range(7):
        line = []
        for col in range(7):
            piece = game_set.board[(row, col)]
            line.append(PIECE_TO_CHAR[piece])
            
        if row == 3:
            if line[3] == '.':
                line[3] = '='
        if row == 0 or row == 6:
            if line[0] == '.':
                line[0] = '='
            if line[-1]== '.':
                line[-1]= '='
                
        print('{0}| {1} |{0}'.format(row, ' '.join(line)))
    print(' +---------------+ ')
    print('   ' + ' '.join(COLS))
    

class Player():
    
    def select_move(self, game_state):
        # Keep asking the user for input until valid input recieved
        move_is_illegal = 'True'
        while move_is_illegal:
            
            # Ask for input
            next_move = input('-- ')
            
            if next_move == 'pass':
                # TODO: Should check if this is legal.
                return Act.pass_turn()
                
            elif next_move == 'resign':
                return Act.resign()
            
            # Check if input expression doesn't identify a move
            elif not re.search("[A-G][0-6]  *[A-G][0-6]", next_move):
                print('Please provide valid input')
                continue
            
            else:
                # If the input does identify a move then use it to create
                # a tuple 'move' representing the move
                piece, point = next_move.split()
                piece = (int(piece[1]), COLS.index(piece[0]))
                point = (int(point[1]), COLS.index(point[0]))
                move = piece + point
                
                # Try make the move and see if it is legal. If not, print
                # why it isn't to the screen and wait the next input
                move_is_illegal = game_state.take_turn(Act.play(move))
                if move_is_illegal:
                    print(move_is_illegal)
                else:
                    return Act.play(move)
    