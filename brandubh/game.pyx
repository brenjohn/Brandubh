# cython: linetrace=True
# cython: language_level=3
"""
Created on Thu Dec 26 13:55:09 2019

@author: john

This file defines classes that can be used to implement a game of brandubh.

classes:
    Act       - representing the actions a player can take during their turn
    GameSet   - representing both the game board and the pieces on the board
    GameState - representing the state of the game for a given board position
"""

from libc.stdint cimport int64_t
from libc.string cimport memcmp

cdef extern from *:
    """
    #define IS_ON_BOARD(r, c) ((0 <= r && r <= 6) && (0 <= c && c <= 6))
    #define IS_HOSTILE_SQUARE(r, c) ((r == 0 || r == 6) && (c == 0 || c == 6))
    #define IS_SPECIAL_SQUARE(r, c) ((r == 3 && c == 3) || IS_HOSTILE_SQUARE(r, c))
    """
    bint IS_ON_BOARD(int r, int c)
    bint IS_HOSTILE_SQUARE(int r, int c)
    bint IS_SPECIAL_SQUARE(int r, int c)


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



cdef class GameSet:
    """
    This class represents the brandubh game set, i.e. the game board and 
    game pieces (soldiers and king)
    
    The 7x7 board is stored as an int c-array of length 7x7. Pieces on the 
    board are numbered 1 to 16 and a zero in this array represents an empty 
    square.
    
    The row and coloumn coordinates of pieces are stored in two int arrays.
    The coordinates of black pieces are stored in even slots of these arrays
    while those for white pieces are stored in odd slots. For technical reasons
    some of the slots in these arrays are left redundant. Below is an outline
    of how the slots are used, this in turn determines how pieces on the board
    are numbered. 
    (Note, N = unused slot, K = king, b = black soldier, w = white soldier)
    
        [N, K, b, w, b, w, b, w, b, w, b, N, b, N, b, N, b]
        
    Pieces that are removed from the board have their coordinates set to -1.
    """
    
    cdef int[49] board
    cdef int[17] piece_row
    cdef int[17] piece_col
        
        
    @classmethod
    def empty_board(cls):
        board      = [0]  * 49
        pieces_row = [-1] * 17
        pieces_col = [-1] * 17
        return GameSet(board, pieces_row, pieces_col)
    
    
    def copy(self):
        """
        Returns a copy of this GameSet.
        """
        return self._copy()
    
    
    cdef GameSet _copy(self):
        cdef GameSet game_set = GameSet()
        game_set.init_board(self.board, self.piece_row, self.piece_col)
        return game_set
    
    
    cdef void init_board(
            self, 
            int[49] board, 
            int[17] piece_row, 
            int[17] piece_col
        ):
        self.board = board
        self.piece_row = piece_row
        self.piece_col = piece_col
    
    
    def game_pieces(self):
        """
        Returns a dictionary representing the board position, the keys of the 
        dictionary are tuples containing the coordinates of the squares on the 
        board and the value of a key is the piece located on that square. The 
        values for the different pieces are:
            The king = 2
            white soldier = 1
            black soldier = -1
        """
        game_pieces = {}
        
        if self.piece_row[1] != -1:
            square = self.piece_row[1], self.piece_col[1]
            game_pieces[square] = 2
            
        for i in range(3, 11, 2):
            if self.piece_row[i] != -1:
                square = self.piece_row[i], self.piece_col[i]
                game_pieces[square] = 1
                
        for i in range(2, 18, 2):
            if self.piece_row[i] != -1:
                square = self.piece_row[i], self.piece_col[i]
                game_pieces[square] = -1
        
        return game_pieces
    
    
    def get_piece(self, int row, int col):
        """
        This method returns the piece on the given square.
        """
        return self._get_piece(row, col)
    
    cdef int _get_piece(self, int row, int col):
        return self.board[row + 7 * col]


    def is_on_board(self, int row, int col):
        """
        This method checks if a given square is on the board
        """
        return IS_ON_BOARD(row, col)
    
    
    def is_hostile_square(self, int row, int col):
        """
        This method checks if a given square is a hostile square
        """
        return IS_HOSTILE_SQUARE(row, col)


    def is_special_square(self, int row, int col):
        """
        This method checks if a given square is a special square
        """
        return IS_SPECIAL_SQUARE(row, col)
    
    
    # Return the position of the given piece.
    cpdef tuple[int, int] piece_position(self, int piece):
        return tuple((self.piece_row[piece], self.piece_col[piece]))


    def move_piece(self, int ir, int ic, int fr, int fc):
        """
        This method implements a move by updating the board variable with
        the new position of a piece and removes captured pieces
        """
        self._move_piece(ir, ic, fr, fc)


    cdef void _move_piece(self, int ir, int ic, int fr, int fc):
        cdef int dr, dc, nr, nc, nnr, nnc
        cdef int piece, neighbour, far_square
        
        # Move the piece to its new location.
        piece = self.board[ir + 7 * ic]
        self.board[fr + 7 * fc] = piece
        self.board[ir + 7 * ic] = 0
        self.piece_row[piece] = fr
        self.piece_col[piece] = fc
        
        # Loop over neighbouring squares, check if any pieces are captured and
        # remove them if they are.
        cdef int n
        cdef int[4] drs = (0, 0, -1, 1)
        cdef int[4] dcs = (-1, 1, 0, 0)
        for n in range(4):
            dr = drs[n]
            dc = dcs[n]
            
            # Get the square on the far side of a neighbouring square and check
            # if it is on the board. If not, continue to the next neighbour.
            nnr, nnc = fr + 2 * dr, fc + 2 * dc
            # TODO: Why doesn't inlining work here?
            if not IS_ON_BOARD(nnr, nnc):
                continue
            
            # Get the neighbouring square.
            nr, nc = fr + dr, fc + dc

            # If neighbouring square does not have an enemy, continue.
            neighbour = self.board[nr + 7 * nc]
            if (neighbour == 0) or (not (neighbour + piece) & 1):
                continue

            # If the far square is a hostile square, capture the enemy piece.
            if IS_HOSTILE_SQUARE(nnr, nnc):
                self.remove_piece(neighbour, nr, nc)
                continue

            # If the far square contains an ally piece, capture the enemy 
            # piece.
            far_square = self.board[nnr + 7 * nnc]
            if far_square != 0 and not ((far_square + piece) & 1):
                self.remove_piece(neighbour, nr, nc)
                    
                    
    cdef void remove_piece(self, int piece, int row, int col):
        """
        This method removes the given piece, located at (row, col), from the 
        board
        """
        self.piece_row[piece] = -1
        self.piece_col[piece] = -1
        self.board[row + 7 * col] = 0


    def set_board(self, list board_position):
        """
        This method sets the board up with a given board position
        """
        self.board = board_position
        self.update_pieces()


    cdef void update_pieces(self):
        """
        This method updates the list of pieces with the current positions of 
        the board pieces
        """
        cdef int row, col, piece
        
        for piece in range(17):
            self.piece_row[piece] = -1
            self.piece_col[piece] = -1
        
        for row in range(7):
            for col in range(7):
                piece = self.board[row + 7 * col]
                if piece > 0:
                    self.piece_row[piece] = row
                    self.piece_col[piece] = col
    
    
    def king_captured(self):
        """
        Check if the king piece has been captured.
        """
        return self._king_captured()
    
    cdef bint _king_captured(self):
        return self.piece_row[1] == -1



cdef class GameState:
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
    
    The history field is the root of a linked list containing the history of
    the current game.
    """
    
    cdef GameSet _game_set
    cdef int _player
    cdef int _winner
    cdef int _num_moves
    cdef HistoryLink _history

    def __init__(self, game_set, player, winner=0, history=None, num_moves=0):
        self._game_set = game_set
        self._player = player
        self._winner = winner
        self._num_moves = num_moves
        if history:
            self._history = history
        else:
            self._history = HistoryLink(game_set, player)
            
    property player:
        def __get__(self):
            return self._player
        
    property winner:
        def __get__(self):
            return self._winner
        
    property history:
        def __get__(self):
            return self._history
        
    property game_set:
        def __get__(self):
            return self._game_set
        
    property num_moves:
        def __get__(self):
            return self._num_moves
    
    
    def game_pieces(self):
        """
        Returns a dictionary representing the board position, the keys of the 
        dictionary are tuples containing the coordinates of the squares on the 
        board and the value of a key is the piece located on that square. The 
        values for the different pieces are:
            The king = 2
            white soldier = 1
            black soldier = -1
        """
        return self._game_set.game_pieces()


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
        cdef int ir, ic, fr, fc
        self._num_moves += 1
        if action.is_play:
            # Move the piece
            ir, ic, fr, fc = action.move
            self._game_set._move_piece(ir, ic, fr, fc)

            # If King moved to corner square, white wins
            if IS_HOSTILE_SQUARE(fr, fc):
                self._winner = 1
            # if action.move[2:] in self.game_set.hostile_squares:
            #     self.winner = 1
 
            # If King is captured, black wins
            if self._game_set._king_captured():
                self._winner = -1

            # The other player makes a move next
            self._player *= -1
            
            # Add the new game state to the game hisory.
            self._history = HistoryLink(self._game_set, 
                                        self._player,
                                        (ir, ic, fr, fc),
                                        self._history)

        elif action.is_pass:
            # If action is a pass, the other player makes a move next
            self._player *= -1
            
        else:
            # If the action is a resignation, the other player wins.
            self._winner = -1*self._player


    @classmethod
    def new_game(cls, board = None):
        """
        This class method returns a GameState object for a new game by
        setting the board and giving the black player the first move.
        
        If a board position is supplied, it will be used as the starting
        position.
        """
        if not board:
            # Create a 2D array for the board
            board = [0] * 49
    
            # Set up the black pieces
            for n, i in enumerate([0, 1, 5, 6]):
                board[3 + 7 * i] = 2 * n + 2
                board[i + 7 * 3] = 2 * n + 10
    
            # Set up the white pieces
            for n, i in enumerate([2, 4]):
                board[3 + 7 * i] = 2 * n + 3
                board[i + 7 * 3] = 2 * n + 7
    
            # Place the king piece in the centre
            board[3 + 7 * 3] = 1

        game_set = GameSet.empty_board()
        game_set.set_board(board)
        return GameState(game_set, -1)


    def is_not_over(self):
        """
        Checks if the game hasn't ended yet.
        """
        return self._winner == 0


    def is_move_illegal(self, move):
        """
        This method checks if a given move violates any game rules.
        """
        # You can't make a move if the game is over.
        if not self.is_not_over():
            return 'The game is over'
        
        ri, ci, rf, cf = move
        game_set = self._game_set
        
        # Check if a piece is being moved to the same square.
        if (ri, ci) == (rf, cf):
            return "You must move a piece to a new square or pass"
        
        piece = game_set.get_piece(ri, ci)
        if piece == 0:
            return "There's no piece there"

        # Check if piece being moved belongs to the player.
        if (piece % 2) != ((self._player + 1) // 2):
            return "This piece doesn't belong to you"

        # Check if a soldier is being moved to a special square.
        if game_set.is_special_square(rf, cf) and piece != 1:
            return 'You cannot move soldiers to special squares'

        # Check for pieces standing between (xi,yi) and (xf,yf).
        valid_move = False
        if ri == rf:
            increment = 1 if ci-cf > 0 else -1
            for i in range(cf, ci, increment):
                if game_set.get_piece(rf, i) != 0:
                    return 'You cannot jump pieces'
            else:
                valid_move = True

        elif ci == cf:
            increment = 1 if ri-rf > 0 else -1
            for i in range(rf, ri, increment):
                if game_set.get_piece(i, cf) != 0:
                    return 'You cannot jump pieces'
            else:
                valid_move = True
            
        # Get the board position if the move was made and check that it hasn't
        # occurred already in the game history.
        if valid_move:
            return self.next_board_already_played(ri, ci, rf, cf)

        # If the 'return None' statement is not reached, the move must be 
        # illegal.
        return 'That is not a valid move'


    def legal_moves(self):
        """
        This method returns a list of legal moves that can be made given the
        current game state.
        """
        cdef int player = self._player
        cdef GameSet game_set = self._game_set
        
        cdef list all_moves = [(-1, -1, -1, -1)] * 96
        cdef list winning_moves = [(-1, -1, -1, -1)] * 96
        
        cdef int num_moves = 0
        cdef int num_winning_moves = 0

        # If the player is white, we want to loop over all white pieces,
        # otherwise we want to loop over all black pieces.
        cdef int first, last
        if player == 1:
            first, last = 1, 11
        else:
            first, last = 2, 18

        # For each piece, we loop through all the squares between the piece and
        # the edge of the board in each of the four directions.
        cdef int[4] directions_r = (0, 0, -1, 1)
        cdef int[4] directions_c = (-1, 1, 0, 0)
        cdef int dr, dc, ir, ic, fr, fc, i, piece
        
        cdef GameSet next_game_set
        cdef bytes next_board
        cdef tuple[int, int, int, int] move
        
        # Loop over the player's pieces.
        for piece in range(first, last, 2):
            ir, ic = game_set.piece_position(piece)
            
            # Make sure current piece is on the board.
            if ir >= 0:
                
                # Loop over the four different directions.
                for n in range(4):
                    dr = directions_r[n]
                    dc = directions_c[n]
                    
                    # Loop over all squares from the current piece to the edge.
                    for i in range(1,6):
                        fr = ir + i * dr
                        fc = ic + i * dc
                        
                        # If the current square is off the board, contains
                        # another piece, or is a special square while the
                        # current piece is not a king, move onto looping over
                        # the next direction
                        if not IS_ON_BOARD(fr, fc):
                            break
                        if game_set._get_piece(fr, fc) != 0:
                            break
                        
                        # TODO: maybe reduce the number of appends by only
                        # having non-winning moves in the moves list?
                        if piece != 1:
                            if IS_SPECIAL_SQUARE(fr, fc):
                                continue
                        else:
                            if IS_HOSTILE_SQUARE(fr, fc):
                                move = (ir, ic, fr, fc)
                                winning_moves[num_winning_moves] = move
                                all_moves[num_moves] = move
                                num_winning_moves += 1
                                num_moves += 1
                                continue
                            
                        next_game_set = self._game_set._copy()
                        next_game_set._move_piece(ir, ic, fr, fc)
                        
                        if next_game_set._king_captured():
                            move = (ir, ic, fr, fc)
                            winning_moves[num_winning_moves] = move
                            all_moves[num_moves] = move
                            num_winning_moves += 1
                            num_moves += 1
                            continue
                        
                        if self._history:
                            if self._history.has_been_played(next_game_set):
                                continue
                        
                        move = (ir, ic, fr, fc)
                        all_moves[num_moves] = move
                        num_moves += 1

        return all_moves[:num_moves], winning_moves[:num_winning_moves]
    
    
    def next_board_already_played(self, int ir, int ic, int fr, int fc):
        """
        Checks if moving the piece at (ir, ic) to (fr, fc) moves the game state
        into a board position which has already occurred in the game. Returns 
        an error message if it does and 'None' if it doesn't.
        """
        cdef GameSet next_game_set
        
        if self._history:
            next_game_set = self._game_set._copy()
            next_game_set._move_piece(ir, ic, fr, fc)
            if self._history.has_been_played(next_game_set):
                return 'You cannot move into a previous board position'
        
        return None
        

    def copy(self):
        """
        Return a copy of the GameState which can be modified without changing
        the original.
        """
        # game_set = copy.deepcopy(self.game_set)
        cdef GameSet game_set = self._game_set._copy()
        return GameState(game_set, 
                         self._player, 
                         self._winner, 
                         self._history, 
                         self._num_moves)



cdef class HistoryLink:
    """
    This link class can be used to store the history of a game in a linked 
    list. Each link in the list records the board position of a turn in the
    game history and which player was making the next move.
    """
    cdef int64_t[3]  _board
    cdef int         _player
    cdef int[4]      _last_move
    cdef HistoryLink _previous_state
    cdef int         _turn
    
    def __init__(self, game_set, player, move=None, previous_state=None):
        get_board_state(self._board, game_set)
        self._player = player
        if move:
            self._last_move = move
        else:
            self._last_move = (-1, -1, -1, -1)
        self._previous_state = previous_state
        self._turn = 0 if previous_state == None else previous_state.turn + 1
        
    property board:
        def __get__(self):
            return self._board
        
    property player:
        def __get__(self):
            return self._player
        
    property last_move:
        def __get__(self):
            if self._last_move[0] == -1:
                return None
            return tuple(self._last_move)
        
    property previous_state:
        def __get__(self):
            return self._previous_state if self._previous_state else None
        
    property turn:
        def __get__(self):
            return self._turn
    
    
    cdef bint has_been_played(self, GameSet game_set):
        # Check if the given board position matches any of the board positions
        # in the game history.
        cdef int64_t[3] board
        get_board_state(board, game_set)
        
        historic_state = self
        while historic_state is not None:
            if memcmp(&board[0], &historic_state._board[0], sizeof(board)) == 0:
                return True
            historic_state = historic_state._previous_state
        else:
            return False
    
    
    def corresponds_to(self, int player, GameSet game_set):
        """
        Checks if the given combination of player and game_set corresponds to
        the same state this history link represents.
        """
        # same_player = player == self.player
        # return same_player and self.board_state(game_set) == self.board
        cdef int64_t[3] board
        
        if player == self.player:
            get_board_state(board, game_set)
            if memcmp(&board[0], &self._board[0], sizeof(board)) == 0:
                return True
        return False
    
    
cdef void get_board_state(int64_t[3] state, GameSet game_set):
    """
    """
    cdef int square, piece
    cdef int64_t mask = 1
    cdef int[49] board = game_set.board
    
    state[0] = state[1] = state[2] = 0
    
    for square in range(49):
        piece = board[square]
        
        if piece > 1:
            state[piece & 1] |= mask
        
        elif piece == 1:
            state[2] |= mask
            
        mask <<= 1