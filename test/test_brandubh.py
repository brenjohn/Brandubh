#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:19:41 2021

@author: john
"""

import unittest
from brandubh.brandubh import Act, GameSet, GameState


class TestAct(unittest.TestCase):

    def test_play(self):
        move = (1, 1, 2, 2)
        act = Act.play(move)
        self.assertTrue(act.is_play, "act should be a play")

    def test_pass_turn(self):
        act = Act.pass_turn()
        self.assertTrue(act.is_pass, "act should be a pass")
        
    def test_resign(self):
        act = Act.resign()
        self.assertTrue(act.is_resign, "act should be a resignation")
        
    
        
class TestGameSet(unittest.TestCase):

    def test_move_piece(self):
        # TODO: write a function to create the statndard board dictionary.
        # Create a dictionary for the board
        board = {}
        for i in range(7):
            for j in range(7):
                board[(i,j)] = 0
        # Set up the black pieces
        for i in [0, 1, 5, 6]:
            board[(3, i)] = -1
            board[(i, 3)] = -1
        # Set up the white pieces
        for i in [2, 4]:
            board[(3,i)] = 1
            board[(i,3)] = 1
        # Place the king piece in the centre
        board[(3,3)] = 2
        
        game_set = GameSet()
        game_set.set_board(board)
        
        game_set.move_piece((0, 3, 0, 1))
        
        self.assertIn((0, 1), game_set.black_pieces, "piece wasn't moved to final position")
        self.assertNotIn((0, 3), game_set.black_pieces, "piece wasn't removed from initial position")
        
        
        
class TestGameState(unittest.TestCase):

    def test_new_game(self):
        game = GameState.new_game()
        self.assertEqual(game.player, -1, "black should move first")
        self.assertEqual(game.winner, 0, "there should be no winner at the start of a game")

if __name__ == '__main__':
    unittest.main()
