#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 20:07:46 2023

@author: john
"""

import unittest
import numpy as np

from brandubh.game import GameState
from brandubh.bots.zero_bot.networks.zero_network import ZeroNet
from brandubh.bots.zero_bot.networks.dual_network import DualNet
from brandubh.bots.zero_bot.brandubh_zero import ZeroBot
        
        
class TestBrandubhZero(unittest.TestCase):

    def setUp(self):
        self.game = GameState.new_game()

    def test_play_ZeroNet(self):
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

        # Test data expansion.
        exp = self.get_dummy_experience()
        X, Y, R = bot.network.create_training_data(exp)
        self.assertTrue(len(X) == len(Y))
        self.assertTrue(len(X) == len(R))
    
    def test_play_DualNet(self):
        net = DualNet()
        bot = ZeroBot(evals_per_turn=140, batch_size=35, network=net)
        loss_weights = (1.0, 0.1)
        bot.compile_network(loss_weights)
        
        move = bot.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")
        
        # Run a second time to test search tree reuse.
        self.game.take_turn_with_no_checks(move)
        move = bot.select_move(self.game)
        self.assertTrue(move, "No move returned by bot")

        # # Test data expansion.
        # exp = self.get_dummy_experience()
        # X, Y, R = bot.network.create_training_data(exp)
        # self.assertTrue(len(X) == len(Y))
        # self.assertTrue(len(X) == len(R))
        
    def get_dummy_experience(self):
        boards = np.array([[[0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.]],

               [[0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.]],

               [[0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 1., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.]],

               [[1., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 1.],
                [0., 0., 1., 0., 0., 1.],
                [0., 0., 1., 1., 0., 1.],
                [0., 0., 1., 0., 0., 1.],
                [1., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 1.]],

               [[0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 1., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.]],

               [[0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.]],

               [[0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 1.]]])

        priors = {(0, 3, 0, 2): 52,
                 (0, 3, 0, 1): 68,
                 (0, 3, 0, 4): 1,
                 (0, 3, 0, 5): 1,
                 (1, 3, 1, 2): 61,
                 (1, 3, 1, 1): 62,
                 (1, 3, 1, 0): 202,
                 (1, 3, 1, 4): 60,
                 (1, 3, 1, 5): 1,
                 (1, 3, 1, 6): 58,
                 (3, 0, 2, 0): 56,
                 (3, 0, 1, 0): 62,
                 (3, 0, 4, 0): 52,
                 (3, 0, 5, 0): 1,
                 (3, 1, 2, 1): 61,
                 (3, 1, 1, 1): 91,
                 (3, 1, 0, 1): 117,
                 (3, 1, 4, 1): 56,
                 (3, 1, 5, 1): 60,
                 (3, 1, 6, 1): 130,
                 (3, 5, 2, 5): 50,
                 (3, 5, 1, 5): 55,
                 (3, 5, 0, 5): 53,
                 (3, 5, 4, 5): 51,
                 (3, 5, 5, 5): 59,
                 (3, 5, 6, 5): 55,
                 (3, 6, 2, 6): 1,
                 (3, 6, 1, 6): 1,
                 (3, 6, 4, 6): 58,
                 (3, 6, 5, 6): 59,
                 (5, 3, 5, 2): 61,
                 (5, 3, 5, 1): 56,
                 (5, 3, 5, 0): 1,
                 (5, 3, 5, 4): 52,
                 (5, 3, 5, 5): 1,
                 (5, 3, 5, 6): 66,
                 (6, 3, 6, 2): 56,
                 (6, 3, 6, 1): 60,
                 (6, 3, 6, 4): 52,
                 (6, 3, 6, 5): 1}

        exp = {'boards'        : [boards,],
               'moves'         : [(3, 1, 2, 1),],
               'prior_targets' : [priors,],
               'players'       : [1,],
               'winner'        : 1}
        
        return [exp,]

if __name__ == '__main__':
    unittest.main()
