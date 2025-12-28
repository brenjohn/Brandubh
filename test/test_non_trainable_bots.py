#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 22:44:58 2023

@author: john
"""

import unittest
from brandubh.game import GameState
from brandubh.bots.random_bot import RandomBot
from brandubh.bots.greedy_random_bot import GreedyRandomBot
from brandubh.bots.mcbot import MCTSBot


class TestRandBot(unittest.TestCase):
    
    def setUp(self):
        self.bot = RandomBot()
        self.game = GameState.new_game()

    def test_play(self):
        move = self.bot.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")
        
    
        
class TestGreedyRandBot(unittest.TestCase):
    
    def setUp(self):
        self.bot1 = GreedyRandomBot()
        self.bot2 = GreedyRandomBot(filter_losing_moves = True)
        self.game = GameState.new_game()

    def test_play(self):
        move = self.bot1.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")
        
        move = self.bot2.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")
        
        
        
class TestMCTSBot(unittest.TestCase):

    def setUp(self):
        self.bot = MCTSBot(num_rounds = 1)
        self.game = GameState.new_game()

    def test_play(self):
        move = self.bot.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")
        
        # Run a second time to test search tree reuse.
        self.game.take_turn_with_no_checks(move)
        move = self.bot.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")

if __name__ == '__main__':
    unittest.main()
