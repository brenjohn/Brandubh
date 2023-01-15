#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:19:41 2021

@author: john
"""

import unittest
from brandubh.game import Act, GameSet, GameState


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
    
    @classmethod
    def setUpClass(cls):
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
        
        cls.game_set = GameSet()
        cls.game_set.set_board(board)
    
    def test_special_square(self):
        point = (0, 0)
        self.assertTrue(self.game_set.is_special_square(point),
                        "The corner (0, 0) should be a special square")
        point = (1, 0)
        self.assertFalse(self.game_set.is_special_square(point),
                        "The corner (1, 0) is not a special square")

    def test_move_piece(self):
        game_set = self.game_set.copy()
        
        # Test piece is moved from initial position to final position.
        game_set.move_piece((1, 3, 1, 0))
        self.assertIn((1, 0), game_set.black_pieces,
                      "piece wasn't moved to final position")
        self.assertNotIn((1, 3), game_set.black_pieces,
                         "piece wasn't removed from initial position")
        
        # Test if piece captured by opponent piece and hostile square.
        game_set.move_piece((2, 3, 2, 0))
        self.assertIn((2, 0), game_set.white_pieces,
                      "piece wasn't moved to final position")
        self.assertNotIn((1, 0), game_set.black_pieces,
                         "piece wasn't removed after being captured")
        
        # Test if king captured by opponent pieces.
        self.assertFalse(game_set.king_captured)
        game_set.move_piece((3, 3, 0, 4))
        game_set.move_piece((5, 3, 0, 5))
        self.assertTrue(game_set.king_captured, "King wasn't captured")
        
        
        
class TestGameState(unittest.TestCase):

    def test_new_game(self):
        game = GameState.new_game()
        self.assertEqual(game.player, -1, "black should move first")
        self.assertEqual(game.winner, 0, "should be no winner at the start")
        
    def test_take_turn(self):
        game = GameState.new_game()
        
        # Test making a legal move.
        action = Act(move=(0, 3, 0, 2))
        message = game.take_turn(action)
        self.assertTrue(message == None)
        
        # Test making illegal moves.
        action = Act(move=(0, 2, 0, 3))
        message = game.take_turn(action)
        expected = "Either there's no piece there or it doesn't belong to you"
        self.assertTrue(message == expected)
        
        action = Act(move=(4, 3, 0, 3))
        message = game.take_turn(action)
        expected = "You cannot jump pieces"
        self.assertTrue(message == expected)
        
        action = Act(move=(4, 3, 4, 3))
        message = game.take_turn(action)
        expected = "You must move a piece to a new square or pass"
        self.assertTrue(message == expected)
        
        action = Act(move=(3, 3, -1, 2))
        message = game.take_turn(action)
        expected = 'That is not a valid move'
        self.assertTrue(message == expected)
        
    def test_legal_moves(self):
        game = GameState.new_game()
        moves = game.legal_moves()
        self.assertTrue(moves)

if __name__ == '__main__':
    unittest.main()
