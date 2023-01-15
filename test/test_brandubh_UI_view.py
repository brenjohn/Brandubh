#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:27:30 2023

@author: john
"""

import unittest

from unittest.mock import Mock, patch
from brandubh.UI.view import BrandubhView
        
        
class TestBrandubhView(unittest.TestCase):
    
    @patch('curses.start_color')
    @patch('curses.init_pair')
    @patch('curses.curs_set')
    @patch('curses.window')
    def test_initialisation(self, mock_stdscr, mock_curs_set, mock_init_pair,
                            mock_start_color):
        BrandubhView(mock_stdscr)
        
        self.assertTrue(mock_stdscr.clear.call_count == 1)
        self.assertTrue(mock_stdscr.refresh.call_count == 1)
        self.assertTrue(mock_init_pair.call_count == 3)
        self.assertTrue(mock_start_color.call_count == 1)
        self.assertTrue(mock_curs_set.call_count == 1)
    
    
    
    @patch('curses.resizeterm')
    @patch('curses.color_pair')
    @patch('curses.start_color')
    @patch('curses.init_pair')
    @patch('curses.curs_set')
    @patch('curses.window')
    def test_draw_main_menu(self, mock_stdscr, mock_curs_set, mock_init_pair,
                            mock_start_color, mock_color_pair,
                            mock_resizeterm):
        mock_stdscr.getmaxyx.side_effect = [(1, 1)]
        view = BrandubhView(mock_stdscr)
        view.draw_main_menu(2)
        
        self.assertTrue(mock_stdscr.clear.call_count == 2)
        self.assertTrue(mock_stdscr.refresh.call_count == 2)
        self.assertTrue(mock_stdscr.attron.call_count == 5)
        self.assertTrue(mock_stdscr.attroff.call_count == 5)
        self.assertTrue(mock_color_pair.call_count == 8)
        self.assertTrue(mock_curs_set.call_count == 1)
    
    
    
    # TODO: Figure out how to test this function without mocking the
    # draw_panel_border method. The problem is that curses doesn't have correct
    # attributes until initialised and we're avoiding initialising curses for
    # tests.
    @patch('brandubh.UI.view.BrandubhView.draw_panel_border')
    @patch('curses.resizeterm')
    @patch('curses.color_pair')
    @patch('curses.start_color')
    @patch('curses.init_pair')
    @patch('curses.curs_set')
    @patch('curses.window')
    def test_draw_player_selection_screen(self, mock_stdscr, mock_curs_set, 
                                          mock_init_pair, mock_start_color,
                                          mock_color_pair, mock_resizeterm,
                                          mock_draw_panel_border):
        mock_stdscr.getmaxyx.side_effect = [(1, 1)]
        view = BrandubhView(mock_stdscr)
        view.draw_player_selection_screen([1, ], 2)
        
        self.assertTrue(mock_draw_panel_border.call_count == 2)
        self.assertTrue(mock_stdscr.clear.call_count == 2)
        self.assertTrue(mock_stdscr.refresh.call_count == 2)
        self.assertTrue(mock_stdscr.attron.call_count == 7)
        self.assertTrue(mock_stdscr.attroff.call_count == 7)
        self.assertTrue(mock_color_pair.call_count == 10)
        self.assertTrue(mock_curs_set.call_count == 1)
    
    
    
    # @patch('brandubh.UI.view.BrandubhView.draw_panel_border')
    # @patch('curses.resizeterm')
    # @patch('curses.color_pair')
    # @patch('curses.start_color')
    # @patch('curses.init_pair')
    # @patch('curses.curs_set')
    # @patch('curses.window')
    # def test_draw_game_screen(self, mock_stdscr, mock_curs_set, 
    #                           mock_init_pair, mock_start_color,
    #                           mock_color_pair, mock_resizeterm,
    #                           mock_draw_panel_border):
    #     mock_stdscr.getmaxyx.side_effect = [(1, 1)]
    #     view = BrandubhView(mock_stdscr)
        
    #     game_pieces = {(1, 1) : -1,
    #                    (1, 2) : 2}
        
    #     view.draw_game_screen(game_pieces, 1, 1, -1, [(1, 1),], [], None)
        
    #     self.assertTrue(mock_draw_panel_border.call_count == 2)
    #     self.assertTrue(mock_stdscr.clear.call_count == 2)
    #     self.assertTrue(mock_stdscr.refresh.call_count == 2)
    #     self.assertTrue(mock_curs_set.call_count == 1)


if __name__ == '__main__':
    unittest.main()