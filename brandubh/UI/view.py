#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 19:59:07 2021

@author: john

This file defines the BrandubhView class which has methods which can be used to
draw menu screens and a game board for playing brandubh.

These methods draw and print to a curses standard screen (stdscr).
"""
import curses
from . import utils

class BrandubhView:
    
    def __init__(self, stdscr):
        # Declaration of constants used to draw different screens of the game.
        self.stdscr
        self.TITLE_LINES = utils.TITLE.split("\n")
        self.TITLE_LENGTH = max([len(line) for line in self.TITLE_LINES])
        self.SUBTITLE = "F\u00e1ilte go brandubh / Welcome to brandubh"

        self.GAME_PIECES = {-1 : " X ",
                             1 : " O ",
                             2 : "qOp"}

        self.PLAYERS = [" User   ",
                        " Rand   ",
                        " GRand  ",
                        " MCBot  ",
                        " Zero   "]

        self.OPTIONS = ["   Play a game   ",
                        "Read the rulebook",
                        "  Exit brandubh  "]

        self.PLAYER_DESCRIPTIONS = ["- Moves are selected by the user.",
                                    "- A bot to randomly select moves.",
                                    "- A greedy random bot.",
                                    "- A Monte Carlo tree search bot.",
                                    "- A reinforcement learning bot."]
        self.init_view()
    
    
    def init_view(self):
        """
        Initialise the colours for curses to use when rendering different
        screens. Also, set the cursor to be invisible.
        """
        # Clear and refresh the screen for a blank canvas
        stdscr = self.stdscr
        stdscr.clear()
        stdscr.refresh()
        
        # Start colors in curses
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_CYAN)
        
        # Set cursor to be invisible.
        curses.curs_set(0)
        
    
    def prepare_screen(self, w, h):
        # Clear and adjust the screen before drawing anything.
        stdscr = self.stdscr
        stdscr.clear()
        
        # Get the height and width of the window and resize the screen if
        # either go below a certain threshold.
        height, width = stdscr.getmaxyx()
        if width < w:
            width = w
            curses.resizeterm(height, width)
        if height < h:
            height = h
            curses.resizeterm(height, width)
        return stdscr, width, height
    
    
    def draw_main_menu(self, option):
        """
        Draw the main menu for the game and highlight the given option. The
        menu screen consists of a large brandubh title and a subtitle. It also
        has options the user can select from, allowing them to play a game,
        read the rules or quit.
        """
        stdscr, width, height = self.prepare_screen(60, 21)
            
        # Work out where to start printing the title so that it is centred.
        TITLE_LENGTH = self.TITLE_LENGTH
        SUBTITLE = self.SUBTITLE
        title_x = int((width // 2) - (TITLE_LENGTH // 2) - TITLE_LENGTH % 2)
        sub_x = int((width // 2) - (len(SUBTITLE) // 2) - len(SUBTITLE) % 2)
        title_y = 2
        
        # Print the main title abd subtitle.
        stdscr.attron(curses.color_pair(2)) 
        stdscr.attron(curses.A_BOLD)
        for i, line in enumerate(self.TITLE_LINES):
            stdscr.addstr(title_y + i, title_x, line[:width-1])
        stdscr.addstr(title_y + i + 1, sub_x, SUBTITLE[:width-1])
        stdscr.attroff(curses.color_pair(2)) 
        stdscr.attroff(curses.A_BOLD)
        
        # Print the available options with the given option highlighted.
        x = int((width // 2) - 8)
        for op, option_string in enumerate(self.OPTIONS):
            colour = 3 if option == op else 1
            stdscr.attron(curses.color_pair(colour))
            stdscr.addstr(title_y + i + 3 + op, x, option_string[:width-1])
            stdscr.attroff(curses.color_pair(colour))
        stdscr.move(0, 0)
        
        # Refresh the screen
        stdscr.refresh()
    
    
    def draw_player_selection_screen(self, players, option):
        """
        Draws a screen where the user can select two players to play a game.
        The user can select themselves as a player or a bot.
        """
        stdscr, width, height = self.prepare_screen(64, 31)
            
        # Centering calculation.
        centre_x = int(width // 2)
        top_y = 4
        
        # Draw two rectangles, the first to encompass the player options and
        # the second to encompass the selected players.
        self.draw_panel_border(stdscr, centre_x-29, top_y, 58, 18)
        self.draw_panel_border(stdscr, centre_x-29, top_y+20, 58, 3)
        
        # Print a header in the first rectangle detailing which side is being
        # selected for.
        if len(players) < 2:
            side_name = "white" if len(players) == 1 else "black"
            header = "Select a player to play as {0}:".format(side_name)
        elif len(players) == 2:
            header = " Press Enter to start the game."
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(top_y+1, centre_x-16, header)
        stdscr.attroff(curses.A_BOLD)
        stdscr.addstr(top_y+18, centre_x-28, "Press backspace to undo.")
        
        # Print the available options with the current option highlighted.
        x = centre_x-28
        y = top_y+2
        for op, option_string in enumerate(self.PLAYERS):
            colour = 3 if option == op else 1
            stdscr.attron(curses.color_pair(colour))
            stdscr.addstr(y + 3 + 3*op, x, option_string)
            stdscr.attroff(curses.color_pair(colour))
            stdscr.addstr(y + 3 + 3*op, x+13, self.PLAYER_DESCRIPTIONS[op])
            
        # Print which options have been selected so far in the second rectangle.
        b = self.PLAYERS[players[0]] if len(players) > 0 else "__"
        w = self.PLAYERS[players[1]] if len(players) > 1 else "__"
        currently_selected = "{0} as black - vs - {1} as white".format(b, w)
        x = int((width // 2) - (len(currently_selected) // 2))
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(top_y+22, x, currently_selected)
        stdscr.attroff(curses.A_BOLD)
        
        # Refresh the screen
        stdscr.refresh()
        
        
    def draw_loading_screen(self):
        """
        Draws a loading screen. Used when loading a neural network.
        """
        stdscr, width, height = self.prepare_screen(64, 31)
            
        # Centering calculation.
        centre_x = int(width // 2)
        top_y = 4
        
        # Draw the main panel displaying the loading message.
        self.draw_panel_border(stdscr, centre_x-29, top_y, 58, 10)    
        stdscr.addstr(top_y+5, centre_x-5, "Loading...")
        
        # Refresh the screen
        stdscr.refresh()
    
    
    def draw_game_screen(self, game_pieces, 
                         cursor_x, cursor_y, 
                         player, next_move, previous_moves,
                         info_message):
        """
        Draws the screen where the user can inspect and interact with a game of 
        brandubh. To draw the game screen, this function expects as arguments:
            
            stdscr               - A curses standard screen to draw to.
            
            game_pieces          - A dict mapping board squares to the pieces
                                   that occupy them.
                          
            (cursor_x, cursor_y) - The location of the cursor used by the user
                                   to select squares.
                                   
            player               - Which player is to make the next move.
            
            next_move            - An array containing squares making up the
                                   next move selected by the user.
            
            previous_moves       - An array of the most recent moves played, to
                                   be displayed on the screen.
            
            info_message         - A message to be printed to the screen
                                   regarding illegal moves etc.
        """
        stdscr, width, height = self.prepare_screen(64, 26)
            
        if not info_message:
            info_message = ""
            
        # Centering calculation.
        centre_x = int(width // 2)
        top_y = 4
        
        # Draw the board, the info panels and the game pieces.
        self.draw_board(stdscr, centre_x, top_y)
        self.draw_left_panel(stdscr, centre_x, top_y, player, *next_move)
        self.draw_right_panel(stdscr, centre_x, top_y, previous_moves)
        self.draw_bottom_panel(stdscr, centre_x, top_y, info_message)
        self.draw_game_pieces(stdscr, centre_x, top_y, game_pieces)
        self.draw_cursor(stdscr, cursor_x, cursor_y, centre_x, top_y)

        # Move the terminal cursor to the origin when drawing is finished and 
        # refresh the screen
        stdscr.move(0, 0)
        stdscr.refresh()
    
    
    def draw_board(self, centre_x, top_y):
        """
        Draws the brandubh game board.
        """
        # Draw the upper board edge.
        stdscr = self.stdscr
        stdscr.move(top_y, centre_x-14)
        stdscr.addch(curses.ACS_ULCORNER)
        for i in range(1, 28):
            if i%4 == 0:
                stdscr.addch(curses.ACS_TTEE)
            else:
                stdscr.addch(curses.ACS_HLINE)
        stdscr.addch(curses.ACS_URCORNER)
        
        # Draw the inner grid of the board.
        stdscr.move(top_y+1, centre_x-14)
        stdscr.addch(curses.ACS_VLINE)
        for i in range(1, 28):
            if i%4 == 0:
                stdscr.addch(curses.ACS_VLINE)
            else:
                stdscr.addch(" ")
        stdscr.addch(curses.ACS_VLINE)
        
        for j in range(1, 7):
            stdscr.move(top_y+2*j, centre_x-14)
            stdscr.addch(curses.ACS_LTEE)
            for i in range(1, 28):
                if i%4 == 0:
                    stdscr.addch(curses.ACS_PLUS)
                else:
                    stdscr.addch(curses.ACS_HLINE)
            stdscr.addch(curses.ACS_RTEE)
            
            stdscr.move(top_y+2*j+1, centre_x-14)
            stdscr.addch(curses.ACS_VLINE)
            for i in range(1, 28):
                if i%4 == 0:
                    stdscr.addch(curses.ACS_VLINE)
                else:
                    stdscr.addch(" ")
            stdscr.addch(curses.ACS_VLINE)
        
        # Draw the lower board edge.
        stdscr.move(top_y+14, centre_x-14)
        stdscr.addch(curses.ACS_LLCORNER)
        for i in range(1, 28):
            if i%4 == 0:
                stdscr.addch(curses.ACS_BTEE)
            else:
                stdscr.addch(curses.ACS_HLINE)
        stdscr.addch(curses.ACS_LRCORNER)
        
        # Draw special squares.
        for sx, sy in [(0, 0), (0, 6), (6, 0), (6, 6), (3, 3)]:
            x, y = self.square_position(sx, sy, centre_x, top_y)
            stdscr.addch(y, x+1, curses.ACS_DIAMOND)
        
        # Draw the board coordinate axes.
        for i, ch in enumerate("ABCDEFG"):
            x, y = self.square_position(0, 0, centre_x, top_y)
            stdscr.addch(y+2*i, x-4, ch)
            
        for i, ch in enumerate("ABCDEFG"):
            x, y = self.square_position(6, 0, centre_x, top_y)
            stdscr.addch(y+2*i, x+6, ch)
            
        for i, ch in enumerate("1234567"):
            x, y = self.square_position(0, 0, centre_x, top_y)
            stdscr.addch(y-2, x+1+4*i, ch)
            
        for i, ch in enumerate("1234567"):
            x, y = self.square_position(0, 6, centre_x, top_y)
            stdscr.addch(y+2, x+1+4*i, ch)
    
    
    def draw_left_panel(self, centre_x, top_y, player, piece=None, target=None):
        """
        Draws the panel to the left of the game board which is used to indicate
        who is making the next move and what the user selected next move is.
        """
        # Draw the panel border.
        stdscr = self.stdscr
        self.draw_panel_border(stdscr, centre_x-31, top_y, 11, 13)
        
        # Print the panel header indicating who is making the next move and
        # body displaying the next move which moves the 'piece' to the 'target'
        # square. Coordinates for the piece and target square are translated
        # into board coordinates.
        player = "White" if player == 1 else "Black"
        stdscr.addstr(top_y+2, centre_x-30, "{0} to".format(player))
        stdscr.addstr(top_y+3, centre_x-30, "move next:")
        stdscr.addstr(top_y+5, centre_x-30, "From: ")
        if piece:
            piece_str = "{0}{1}".format("ABCDEFG"[piece[1]], piece[0]+1)
            stdscr.addstr(piece_str)
        else:
            stdscr.addch(curses.ACS_S9)
            stdscr.addch(curses.ACS_S9)
        stdscr.addstr(top_y+7, centre_x-30, "To:   ")
        if target:
            target_str = "{0}{1}".format("ABCDEFG"[target[1]], target[0]+1)
            stdscr.addstr(target_str)
        else:
            stdscr.addch(curses.ACS_S9)
            stdscr.addch(curses.ACS_S9)
    
    
    def draw_right_panel(self, centre_x, top_y, previous_moves):
        """
        Draws the panel on the right of the board which is used to display the
        most recent moves of the game.
        """
        # Draw the panel border.
        stdscr = self.stdscr
        self.draw_panel_border(stdscr, centre_x+20, top_y, 11, 13)
        
        # Print a header indicating the format of the printed move 
        # (ie move number: move in board coordinates)
        stdscr.addstr(top_y+1, centre_x+21, "No: Move")
        
        # Print the most recent moves in the array 'previous_moves'
        for i, (turn, move) in enumerate(previous_moves):
            coord1 = "ABCDEFG"[move[1]]
            coord2 = move[0] + 1
            coord3 = "ABCDEFG"[move[3]]
            coord4 = move[2] + 1
            move = "{0}{1}->{2}{3}".format(coord1, coord2, coord3, coord4)
            move_str = "{0}: {1}".format(turn, move)
            stdscr.addstr(top_y+3+2*i, centre_x+21, move_str)
    
       
    def draw_bottom_panel(self, centre_x, top_y, message = ""):
        """
        Draws the panel below the board which is used to print messages
        regarding illegal moves and to indicate special keys the user can press
        to pass a turn or end the game.
        """
        # Draw the panel border and print the relevant info.
        stdscr = self.stdscr
        self.draw_panel_border(stdscr, centre_x-31, top_y+16, 62, 3)
        stdscr.addstr(top_y+17, centre_x-29, message)
        stdscr.addstr(top_y+19, centre_x-29, "p := pass")
        stdscr.addstr(top_y+19, centre_x-5, "r := resign")
        stdscr.addstr(top_y+19, centre_x+21, "q := quit")
    
    
    def draw_panel_border(self, x, y, w, h):
        """
        Draws a rectangle whose upper left corner is located at (x, y) and
        whose width and height are 'w' and 'h' respectively.
        """
        stdscr = self.stdscr
        stdscr.move(y, x)
        stdscr.addch(curses.ACS_ULCORNER)
        for i in range(1, w):
            stdscr.addch(curses.ACS_HLINE)
        stdscr.addch(curses.ACS_URCORNER)
        
        stdscr.move(y+h+1, x)
        stdscr.addch(curses.ACS_LLCORNER)
        for i in range(1, w):
            stdscr.addch(curses.ACS_HLINE)
        stdscr.addch(curses.ACS_LRCORNER)
        
        for i in range(h):
            stdscr.move(y+1+i, x)
            stdscr.addch(curses.ACS_VLINE)
            stdscr.move(y+1+i, x+w)
            stdscr.addch(curses.ACS_VLINE)
    
    
    def draw_game_pieces(self, centre_x, top_y, game_pieces):
        """
        Draws the game pieces in the locations indicated by the dict
        'game_pieces'.
        """
        stdscr = self.stdscr
        stdscr.attron(curses.A_BOLD)
        for position, piece in game_pieces.items():
            if piece != 0: # Skip empty squares. 
                square_x, square_y = position
                x, y = self.square_position(square_x, square_y, centre_x, top_y)
                piece_symbol = self.GAME_PIECES[piece]
                colour = 1 if piece < 0 else 2
                stdscr.attron(curses.color_pair(colour))
                stdscr.addstr(y, x, piece_symbol)
                stdscr.attroff(curses.color_pair(colour))
        stdscr.attroff(curses.A_BOLD)
    
    
    def draw_cursor(self, cursor_x, cursor_y, centre_x, top_y):
        """
        Highlights the board square indicated by (cursor_x, cursor_y)
        """
        stdscr = self.stdscr
        x, y = self.square_position(cursor_x, cursor_y, centre_x, top_y)
        for i in range(3):
            ch_attrs = stdscr.inch(y, x + i)
            ch = chr(ch_attrs & 0xFF)
            stdscr.attron(curses.color_pair(3))
            stdscr.addch(y, x + i, ch)
            stdscr.attroff(curses.color_pair(3))
    
    
    def draw_game_over_banner(self, winner):
        """
        Draws a rectangle above the board stating the game is over and prints
        who won the game.
        """
        stdscr = self.stdscr
        # Work out the centre of the screen.
        height, width = stdscr.getmaxyx()
        if width < 64:
            width = 64
            curses.resizeterm(height, width)
        centre_x = int(width // 2)
        
        # Draw the 'game over' banner.
        self.draw_panel_border(stdscr, centre_x-12, 1, 24, 1)
        winner_name = "White" if winner == 1 else "Black"
        stdscr.attron(curses.A_BOLD)
        stdscr.addstr(2, centre_x-10, "Game Over: {0} Wins".format(winner_name))
        stdscr.attroff(curses.A_BOLD)
    
    
    def square_position(self, square_x, square_y, centre_x, top_y):
        """
        Converts square coordinates to standard screen coordinates.
        """
        x = centre_x - 14 + 4*square_x + 1
        y = top_y + 2*square_y + 1
        return x, y


    def draw_rulebook(self, page):
        """
        Draws a screen where the user can read the rules of the game.
        """
        stdscr, width, height = self.prepare_screen(64, 31)
            
        # Centering calculation.
        centre_x = int(width // 2)
        top_y = 4
        
        # Draw the main panel displaying the rules.
        self.draw_panel_border(stdscr, centre_x-29, top_y, 58, 20)    
        for i, line in enumerate(utils.RULEBOOK_PAGES[page].splitlines()):
            stdscr.addstr(top_y+i, centre_x-28, line)
        stdscr.addstr(top_y+21, centre_x-6, " Page {0} of 8 ".format(page))
        
        # Draw the bottom panel with the key legend.
        self.draw_panel_border(stdscr, centre_x-29, top_y+22, 58, 1)
        stdscr.addstr(top_y+23, centre_x-28, "<- := previous")
        stdscr.addstr(top_y+23, centre_x-5, "-> := next")
        stdscr.addstr(top_y+23, centre_x+20, "q := quit")
        
        # Refresh the screen
        stdscr.refresh()