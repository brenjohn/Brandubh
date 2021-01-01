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

RULEBOOK = """
Brandubh (Black Raven in Irish) is a small Tafl game known to have been played 
in Ireland at least a thousand years ago.

It is a two player game, played on a 7 x 7 board with the corners and centre 
square marked as special squares (marked as '=' on the board).

The white player begins the game with four soldier pieces (marked as 'o') and
a king (marked as 'A'). The black player starts with 8 soldier pieces (marked
as 'x'). The game begins with the pieces arranged as follows:
    
                           A B C D E F G
                         +---------------+
                        0| = . . x . . = |0
                        1| . . . x . . . |1
                        2| . . . o . . . |2
                        3| x x o A o x x |3
                        4| . . . o . . . |4
                        5| . . . x . . . |5
                        6| = . . x . . = |6
                         +---------------+ 
                           A B C D E F G

The aim of the game for the white player is to move their king to one of the
special corner squares. For the black player, the goal is to capture the king.

All the pieces move the same way: in straight lines forward, backwards, left 
or right, any number of squares without jumping over other pieces.

You can capture an enemy piece by surrounding it on both sides with two of 
your own pieces. They must be standing on opposite sides, either in front and 
behind, or to the left and right, not diagonally. A piece that becomes trapped 
like this by an enemy move is captured and removed from the board. However, it
is safe to move into a gap between two enemies without being captured. The 
King may participate in capturing, just like any other piece, and may also be 
captured just like any other piece. It is possible to capture more than one 
piece at a time, but not if they are standing together in a row.

The central square (known as the throne square) and the four corner squares 
are restricted. Only the king may occupy any of these five squares, though any 
soldier may pass through the throne square when it is empty, without stopping 
on it. The king may return to the throne square after it has left it, if 
required. The king can be captured while on the throne, just the same as on 
any other square, by being surrounded on two opposite sides.

In addition, the four corner squares (but not the throne square) are hostile 
squares. This means that they can play the part of an enemy soldier of either 
colour for the purposes of capturing. Any piece, including the king, that is 
occupying a square next to the corner square, can be captured if an enemy 
piece moves in behind it, trapping it against the hostile corner square.

The game may end in a draw if: (1) either player is unable to move on their 
turn, because all remaining pieces are blocked in and unable to move, (2) a 
perpetually repeating series of moves means the game has reached stalemate by 
repetition, (3) both players agree to a draw at any time.
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
    