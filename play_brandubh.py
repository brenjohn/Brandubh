#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 23:38:28 2023

@author: john

This script creates a UI to allow the user to create and start a game of
brandubh. It is responsibile for handling user input, passing instructions to
both the model (ie the GameState of the game and any bots playing), to progress
the game, and the view, to render information on the terminal screen.

This script uses a curses standard screen (stdscr) to get user input and calls
functions imported from 'brandubh_view.py' to render information on it.
"""
import curses
from brandubh.UI.controller import BrandubhController


def main():
    curses.wrapper(BrandubhController)
    print('Sl\u00e1n / Goodbye')

if __name__ == "__main__":
    main()