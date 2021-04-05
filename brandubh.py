#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:55:09 2019

@author: john

This file defines classes that can be used to implement a game of brandubh.

classes:
    Act       - representing the actions a player can take during their turn
    GameSet   - representing both the game board and the pieces on the board
    GameState - representing the state of the game for a given board position
"""



class Act:
    """
    This class represents the actions a player can take during their turn and
    provides class methods for creating instances of the class for particular
    actions.
    
    On any given turn a player can:
    1) play a move which is represented by a tuple (xi, yi, xf, yf) holding
       the xy coordinates of the piece being moved, (xi, yi) and the xy
       coordinates of the position the piece is being moved to (xf, yf)
       
    2) pass their turn and let the other player take a turn.
    
    3) resign from the game.
    """

    def __init__(self, move=None, is_pass=False, is_resign=False):
        
        # This class must be initialised with one and only one of the three
        # actions that can be taken by a player.
        assert (move is not None) ^ is_pass ^ is_resign
        
        self.move = move
        self.is_play = (self.move is not None)
        self.is_pass = is_pass
        self.is_resign = is_resign


    @classmethod
    def play(cls, move):
        return Act(move=move)


    @classmethod
    def pass_turn(cls):
        return Act(is_pass=True)


    @classmethod
    def resign(cls):
        return Act(is_resign=True)



class GameSet:
    """
    This class represents the brandubh game set, i.e. the game board and 
    game pieces (soldiers and king)
    
    The board is stored as a dictionary, the keys of the dictionary are
    tuples containing the coordinates of the squares on the board and the
    value of a key is the piece located on that square. The values for the 
    different pieces are:
        The king = 2
        white soldier = 1
        black soldier = -1
        
    Lists of the positions of the white and black pieces are also kept for
    convenience when a GameState object is calculating legal moves 
    """

    def __init__(self):
        self.board = {}
        self.special_squares = [(0,0), (0,6), (6,0), (6,6), (3,3)]
        self.hostile_squares = [(0,0), (0,6), (6,0), (6,6)]
        self.neighbours = [(-1,0), (1,0), (0,-1), (0,1)]
        self.white_pieces = []
        self.black_pieces = []
        self.king_captured = None


    def is_on_board(self, square):
        """
        This method checks if a given square is on the board
        """
        row, col = square
        return 0 <= row <= 6 and 0 <= col <= 6


    def is_special_square(self, point):
        """
        This method checks if a given square is a special square
        """
        return point in self.special_squares


    def move_piece(self, move):
        """
        This method implements a move by updating the board variable with
        the new position of a piece and removes captured pieces
        """

        # Get the piece being moved and make sure it exists
        initial_point, final_point = move[:2], move[2:]
        piece = self.board[initial_point]
        assert piece != 0

        #  move the piece to its new location
        self.board[final_point] = piece
        self.board[initial_point] = 0
        if piece > 0:
            self.white_pieces.remove(initial_point)
            self.white_pieces.append(final_point)
        else:
            self.black_pieces.remove(initial_point)
            self.black_pieces.append(final_point)

        # check for captured enemy pieces and remove them
        for neighbour in self.neighbours:

            neighbour_point = add(final_point, neighbour)
            if not self.is_on_board(neighbour_point):
                continue

            #  If neighbouring piece not an enemy, continue
            if self.board[neighbour_point]*piece >= 0:
                continue

            # Get the point on the far side of a neighbouring enemy piece
            far_point = add(final_point, neighbour, const=2)

            # If the far square is a hostile square,
            # capture the enemy piece
            if far_point in self.hostile_squares:
                    
                    # Check if captured piece is a king
                    neighbouring_piece = self.board[neighbour_point]
                    if neighbouring_piece == 2:
                        self.king_captured = True
                        
                    # remove piece
                    if neighbouring_piece > 0:
                        self.white_pieces.remove(neighbour_point)
                    else:
                        self.black_pieces.remove(neighbour_point)
                    self.board[neighbour_point] = 0
                    continue

            # If the far square is on the board and contains 
            # an allied piece, capture the enemy piece
            if self.is_on_board(far_point):
                if self.board[far_point]*piece > 0:
    
                    # Check if captured piece is a king
                    neighbouring_piece = self.board[neighbour_point]
                    if neighbouring_piece == 2:
                        self.king_captured = True
                    
                    #  remove piece
                    if neighbouring_piece > 0:
                        self.white_pieces.remove(neighbour_point)
                    else:
                        self.black_pieces.remove(neighbour_point)
                    self.board[neighbour_point] = 0


    def set_board(self, board_position):
        """
        This method sets the board up with a given board position
        """
        self.board = board_position
        self.update_pieces()
        self.king_captured = 2 not in list(board_position.values())


    def update_pieces(self):
        """
        This method updates the lists white_pieces and black_pieces
        with the current positions of the board pieces
        """
        self.white_pieces = [key for key, val in self.board.items() if val > 0]
        self.black_pieces = [key for key, val in self.board.items() if val < 0]



class GameState:
    """
    This class represents the state of the game for a given board position 
    and has methods for checking the rules of the game and verifying moves
    are legal. It also has methods for starting a new game and taking turns.
    
    The game_set variable is a GameSet object representing the current board
    position
    
    The player variable indicates which player is to make the next move
    (-1 for black and 1 for white)
    
    The winner variable is 0 while the game is not over and set to -1 or 1
    when the game is over depending on who wins
    """

    def __init__(self, game_set, player):
        self.game_set = game_set
        self.player = player
        self.winner = 0
        self.history = HistoryLink(game_set.board, player)


    def take_turn(self, action):
        """
        This method updates a GameState object by applying the move
        represented by the given Act object 'action'. The rules of the game
        are checked before the move is made to make sure it is legal.
        """
        move_illegal = None
        if action.is_play:
            move_illegal = self.is_move_illegal(action.move)
            
        if not move_illegal:
            self.take_turn_with_no_checks(action)
        else:
            return move_illegal
    
    
    def take_turn_with_no_checks(self, action):
        """
        This method updates a GameState object by applying the move
        represented by the given Act object 'action'.
        """
        if action.is_play:
            # Move the piece
            self.game_set.move_piece(action.move)

            # If King moved to corner square, white wins
            if action.move[2:] in self.game_set.hostile_squares:
                self.winner = 1
 
            # If King is captured, black wins
            if self.game_set.king_captured:
                self.winner = -1

            # The other player makes a move next
            self.player *= -1
            
            # Add the new game state to the game hisory.
            self.history = HistoryLink(self.game_set.board, 
                                       self.player,
                                       action.move,
                                       self.history)

        elif action.is_pass:
            # If action is a pass, the other player makes a move next
            self.player *= -1
            
        else:
            # If the action is a resignation, the other player wins.
            self.winner = -1*self.player


    @classmethod
    def new_game(cls, board = None):
        """
        This class method returns a GameState object for a new game by
        setting the board and giving the black player the first move.
        
        If a board position is supplied, it will be used as the starting
        position.
        """
        if not board:
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

        game_set = GameSet()
        game_set.set_board(board)
        return GameState(game_set, -1)


    def is_not_over(self):
        return self.winner == 0


    def is_move_illegal(self, move):
        """
        This method checks if a given move violates any game rules.
        """

        # You can't make a move if the game is over
        if not self.is_not_over():
            return 'The game is over'

        initial_point, final_point = move[:2], move[2:]
        game_set = self.game_set
        
        # Check if a piece is being moved to the same square
        if initial_point == final_point:
            return "You must move a piece to a new square or pass"

        # Check if piece being moved belongs to the player
        if self.player * game_set.board[initial_point] <= 0:
            return "Either there's no piece there or it doesn't belong to you"

        # Check if a soldier is being moved to a special square
        if game_set.is_special_square(final_point) and \
            game_set.board[initial_point] != 2:
            return 'You cannot move soldiers to special squares'

        # Check for pieces standing between (xi,yi) and (xf,yf)
        valid_move = False
        xi, yi, xf, yf = move
        if xi == xf:
            increment = 1 if yi-yf > 0 else -1
            for i in range(yf, yi, increment):
                if game_set.board[(xf, i)] != 0:
                    return 'You cannot jump pieces'
            else:
                valid_move = True

        if yi == yf:
            increment = 1 if xi-xf > 0 else -1
            for i in range(xf, xi, increment):
                if game_set.board[(i, yf)] != 0:
                    return 'You cannot jump pieces'
            else:
                valid_move = True
            
        # Get the board position if the move was made and check that it hasn't
        # occurred already in the game history.
        if valid_move:
            return self.moves_into_previous_board_position(initial_point, 
                                                           final_point)

        # If the 'return None' statement is not reached, the move must be 
        # illegal
        return 'That is not a valid move'


    def legal_moves(self):
        """
        This method returns a list of legal moves that can be made given
        the current game state
        """
        player = self.player
        game_set = self.game_set
        moves = []

        # If the player is white, we want to loop over all white pieces,
        # otherwise we want to loop over all black pieces
        if player == 1:
            pieces = game_set.white_pieces
        else:
            pieces = game_set.black_pieces

        # For each piece we loop through all the squares between the piece and
        # the edge of the board in each of the four directions (neighbours)
        for piece in pieces:
            for neighbour in game_set.neighbours:
                for i in range(1,6):
                    new_sq = add(piece, neighbour, const=i)
                    
                    # If the current square is off the board, contains
                    # another piece, or is a special square while the
                    # current piece is not a king, move onto looping over
                    # the next direction
                    if not game_set.is_on_board(new_sq):
                        break
                    if game_set.board[new_sq] != 0:
                        break
                    if new_sq in game_set.special_squares and \
                        game_set.board[piece] != 2:
                        break
                    if self.moves_into_previous_board_position(piece, new_sq):
                        continue
                    moves.append(piece + new_sq)

        return moves
    
    
    def moves_into_previous_board_position(self, initial_point, final_point):
        """
        Checks if the moving the piece at 'initial_point' to 'final_point'
        moves the game state into a board position which as already occurred
        in the game. Returns an error message if it does and 'None' if it
        doesn't.
        """
        # Create the board position the game would be in if the  specified 
        # move was made.
        next_board = self.game_set.board.copy()
        piece = next_board[initial_point]
        next_board[final_point] = piece
        next_board[initial_point] = 0
        
        # Check if the above board position matches any of the board positions
        # in the game history.
        historic_state = self.history
        while not historic_state == None:
            if next_board == historic_state.board:
                return 'You cannot move into a previous board position'
            historic_state = historic_state.previous_state
        else:
            return None
    


class HistoryLink:
    """
    This link class can be used to store the history of a game in a linked 
    list. Each link in the list records the board position of a turn in the
    game history and which player was making the next move.
    """
    
    def __init__(self, board, player, move=None, previous_state=None):
        self.board = board.copy()
        self.player = player
        self.last_move = move
        self.previous_state = previous_state
        self.turn = 0 if previous_state == None else previous_state.turn + 1



# A convenience function for adding the components of tuples together
def add(tuple_A, tuple_B, const=1):
    return (tuple_A[0]+const*tuple_B[0], tuple_A[1]+const*tuple_B[1])
