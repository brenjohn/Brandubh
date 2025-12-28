#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 20:13:43 2023

@author: john
"""
import unittest
import curses

from unittest.mock import Mock, patch
from brandubh.UI.controller import BrandubhController
        
        
class TestBrandubhController(unittest.TestCase):

    @patch('brandubh.UI.controller.BrandubhController.open_rulebook')
    @patch('brandubh.UI.controller.BrandubhController.start_game')
    @patch('brandubh.UI.view.BrandubhView')
    @patch('curses.window')
    def test_main_menu(self, mock_stdscr, mock_view,
                       mock_start_game, mock_open_rulebook):
        # Test play option.
        mock_stdscr.getch = Mock()
        mock_input = [curses.KEY_DOWN, curses.KEY_DOWN, curses.KEY_DOWN, 10,
                      curses.KEY_DOWN, curses.KEY_DOWN, 10]
        mock_stdscr.getch.side_effect = mock_input
        controller = BrandubhController(None)
        controller.stdscr = mock_stdscr
        controller.view = mock_view
        controller.main_menu()
        self.assertTrue(mock_stdscr.getch.call_count == 7, 'incorrect input')
        self.assertTrue(mock_start_game.call_count == 1, 'game not started')
        
        # Test rulebook option.
        mock_stdscr.getch = Mock()
        mock_input = [curses.KEY_DOWN, 10, curses.KEY_UP, curses.KEY_DOWN,
                      curses.KEY_DOWN, 10]
        mock_stdscr.getch.side_effect = mock_input
        controller = BrandubhController(None)
        controller.stdscr = mock_stdscr
        controller.view = mock_view
        controller.main_menu()
        self.assertTrue(mock_stdscr.getch.call_count == 6, 'incorrect input')
        self.assertTrue(mock_open_rulebook.call_count == 1, 'rulebook not opened')
        
        # Test exit option.
        mock_stdscr.getch = Mock()
        mock_stdscr.getch.side_effect = [curses.KEY_DOWN, curses.KEY_DOWN, 10]
        controller = BrandubhController(None)
        controller.stdscr = mock_stdscr
        controller.view = mock_view
        controller.main_menu()
        self.assertTrue(mock_stdscr.getch.call_count == 3, 'incorrect input')
    
    
    
    @patch('brandubh.UI.view.BrandubhView')
    @patch('curses.window')
    def test_select_players(self, mock_stdscr, mock_view):
        mock_stdscr.getch = Mock()
        mock_input = [10, curses.KEY_DOWN, curses.KEY_DOWN, curses.KEY_DOWN,
                      10, 127, curses.KEY_UP, 10, 10]
        mock_stdscr.getch.side_effect = mock_input
        
        controller = BrandubhController(None)
        controller.stdscr = mock_stdscr
        controller.view = mock_view
        
        controller.select_players()
        self.assertTrue(mock_stdscr.getch.call_count == 9, 'incorrect input')
    
    
    
    @patch('brandubh.game.GameState')
    @patch('brandubh.UI.view.BrandubhView')
    @patch('curses.window')
    def test_play_game(self, mock_stdscr, mock_view, mock_GameState):
        mock_stdscr.getch = Mock()
        mock_input = [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, 10, 127,
                      curses.KEY_RIGHT, ord('p'), ord('r'), ord('q')]
        mock_stdscr.getch.side_effect = mock_input
        # mock_bot = Mock()
        
        mock_game = Mock()
        # mock_game.history = Mock()
        # mock_game.history.last_move = None
        # mock_game.game_set = Mock()
        # mock_game.game_set.board = None
        # mock_game.player.side_effect = 3 * 6 * [-1] + 3 * 2 * [1] + 3 * [-1]
        mock_game.is_not_over.side_effect = [True]
        mock_GameState.new_game = Mock(return_vale = mock_game)
        
        controller = BrandubhController(None)
        controller.stdscr = mock_stdscr
        controller.view = mock_view
        
        controller.play_game("user", "user")
        self.assertTrue(mock_stdscr.getch.call_count == 9, 'incorrect input')
        
    
    
    @patch('brandubh.UI.view.BrandubhView')
    @patch('curses.window')
    def test_play_open_rulebook(self, mock_stdscr, mock_view):
        mock_stdscr.getch = Mock()
        mock_input = [curses.KEY_LEFT, curses.KEY_RIGHT, ord('q')]
        mock_stdscr.getch.side_effect = mock_input
        
        controller = BrandubhController(None)
        controller.stdscr = mock_stdscr
        controller.view = mock_view
        
        controller.open_rulebook()
        self.assertTrue(mock_stdscr.getch.call_count == 3, 'incorrect input')


if __name__ == '__main__':
    unittest.main()
