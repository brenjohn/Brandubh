#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 20:21:37 2023

@author: john
"""
import curses
from .view import BrandubhView

class BrandubhController:
    
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.view = BrandubhView(stdscr)
        self.main_menu()
        
    def main_menu(self):
        """
        This function controls the main menu screen where the user can select to
        play a game, read the rules of the game or exit the game. It also
        initialises curses variables that are used for the duration of the game.
        """
        # Initialise the key variable recording the last key entered by the
        # user and the current option from the main menu.
        key = 0
        option = 0

        # Infinite loop to repeatedly ask the user to enter a new key.
        while True:
            # If the user hit the 'enter' key, select the choosen option,
            if key == 10:
                if option == 0:
                    self.start_game()
                    key = 0
                    continue
                
                # If the chosen option was to read the rules, open rulebook.
                if option == 1:
                    self.open_rulebook()
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
            self.view.draw_main_menu(option)

            # Wait for next input from the user.
            key = self.stdscr.getch()