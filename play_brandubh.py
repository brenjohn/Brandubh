#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 23:38:28 2023

@author: john

This script creates a BrandubhController object which will allow the user to
play a game of brandubh against various bots and read the rules of the game.

The python package curses is used to get user input and render information to
the terminal.
"""
import curses
from brandubh.UI.controller import BrandubhController


def main():
    """
    Create a BrandubhController object which starts the application.
    """
    curses.wrapper(BrandubhController)
    print('Sl\u00e1n / Goodbye')

if __name__ == "__main__":
    main()