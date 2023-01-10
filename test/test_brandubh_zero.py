#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:07:46 2023

@author: john
"""

import unittest

from brandubh.brandubh import GameState
from brandubh.bots.zero_bot.networks.zero_network import ZeroNet
from brandubh.bots.zero_bot.brandubh_zero import ZeroBot
        
        
class TestBrandubhZero(unittest.TestCase):

    def setUp(self):
        self.game = GameState.new_game()

    def test_play(self):
        net = ZeroNet()
        bot = ZeroBot(evals_per_turn=140, batch_size=35, network=net)
        loss_weights = (1.0, 0.1)
        bot.compile_network(loss_weights)
        
        move = bot.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")
        
        # Run a second time to test search tree reuse.
        self.game.take_turn_with_no_checks(move)
        move = bot.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")

    # def test_data_expansion(self):
    #     exp = gain_experience(bot, bot, num_episodes, move_limit, eps)
    #     training_data = bot.network.create_training_data(exp)

if __name__ == '__main__':
    unittest.main()
