#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 16:04:57 2026

@author: john
"""

import numpy as np


class SixPlaneEncoder():
    """
    This class is used to encode a brandubh game state as a tensor which can
    be fed into the neural network created by the ZeroNet class. It also has
    methods for decoding the output tensor from the policy head of the network
    into a dictionary of move-prior pairs, encoding a prior distribution as
    a tensor with the same shape as the policy head output (used for creating
    training data), and for expanding a training data set using symmetries of
    the game board.
    """
        
    def encode(self, game_state):
        """
        A game state is encoded as six 7x7 planes (or a tensor with shape 
        (7,7,6)) as described below.
        
        The first 7x7 plane is an array of 0's and 1's encoding the positions 
        of all pieces owned by the current player, i.e. the array has a 1 in 
        entries corresponding to squares of the board occupied by one of the 
        current players pieces and a 0 otherwise.
        
        The second 7x7 plane is an array encoding the position of king piece 
        owned by the current player. If the player doesn't own a king then 
        this plane will be all zeros, otherwise it will have a 1 in the 
        relevant entry.
        
        The next two planes are the same as the first two but of the opposite 
        player.
        
        The fifth plane is a 7x7 array of 1's if the current player is playing 
        as white and is all 0's otherwise.
        
        The sixth plane is a 7x7 array of 1's if the current player is playing 
        as black and is all 0's otherwise.
        """
        board_tensor = np.zeros((7,7,6))
        player = game_state.player
        game_set = game_state.game_set
        
        if player == 1:
            board_tensor[:, :, 4] = 1
        else:
            board_tensor[:, :, 5] = 1
        
        # white soldier pieces
        for piece in range(1, 11, 2):
            r, c = game_set.piece_position(piece)
            if r > -1:
                board_tensor[r, c, 1 - player] = 1
            
        # King piece
        r, c = game_set.piece_position(1)
        if r > -1:
            board_tensor[r, c, 2 - player] = 1
        
        # black soldier pieces
        for piece in range(2, 18, 2):
            r, c = game_set.piece_position(piece)
            if r > -1:
                board_tensor[r, c, 1 + player] = 1
            
        return board_tensor
    
    def decode_policy(self, model_output, legal_moves):
        """
        The policy head of the ZeroNet outputs a tensor with shape (7,7,24)
        containing a probaility distribution over possible moves to make.
        
        The first two indices of the tensor correspond to a square on the 
        board and the third index indicates a possible move that a piece at 
        that square could possibly make.
        
        Values of the third index ranging from 0 to 5 correspond to decreasing
        the y coordinate of the piece by 6 to 1 places respectively.
        
        Values of the third index ranging from 6 to 11 correspond to increasing
        the y coordinate of the piece by 1 to 6 places respectively.
        
        Values of the third index ranging from 12 to 23 similarly correspond to
        either decreasing or increasing the x coordinate of the piece.
        
        Eg. the prior for moving the piece at square (3, 2) 2 places to (3, 4)
        is output_tensor[3, 2, 7].
        """
        move_priors = {}
        N = 0
        
        for (xi, yi, xf, yf) in legal_moves:
            if yf < yi:
                n = 6 - (yi - yf)
            elif yf > yi:
                n = 5 + (yf - yi)
            elif xf < xi:
                n = 18 - (xi - xf)
            elif xf > xi:
                n = 17 + (xf - xi)
            
            prior = model_output[xi, yi, n]
            move_priors[(xi, yi, xf, yf)] = prior
            N += prior
            
        for move in move_priors.keys():
            move_priors[move] /= N
            
        return move_priors
    
    def encode_prior(self, move_probs):
        target_tensor = np.zeros((7,7,24))
        
        N = 0  # Normalising constant
        for move, prob in move_probs.items():
            xi, yi, xf, yf = move
            if xi == xf:
                k = yf - yi
                k = k + 6 if k < 0 else k + 5
            else:
                k = xf - xi
                k = k + 18 if k < 0 else k + 17
            target_tensor[xi, yi, k] = prob
            N += prob
        
        return target_tensor if N == 0 else target_tensor/N
    
    def encode_priors(self, priors):            
        encoded_priors = [self.encode_prior(prior) for prior in priors]
        num_moves = len(encoded_priors)
        return np.reshape(encoded_priors, (num_moves, 7, 7, 24))
    
    def create_training_data(self, experience):
        """
        A method to convert game data in an experience list to training data
        for training the ZeroBot neural network. The training data is also
        expanded 8 fold using symetries of the game. 
        """
        # The network input and labels forming the training set will be stored 
        # in the following lists.
        X, Y, rewards = [], [], []
        
        # For each episode in the experience append the relevant tensors to 
        # the X, Y and reward lsits.
        for episode in experience:
            Xi = np.array(episode['boards'])
            X.append(Xi)
            
            visit_counts = episode['visit_counts']
            policy_targets = self.encode_priors(visit_counts)
            
            # The reward for moves decays exponentially with the number of
            # moves between it and the winning move. Rewards for moves made by 
            # the winning side are positive and negative for the losing side.
            episode_rewards = episode['winner'] * np.array(episode['players'])
            rewards.append(episode_rewards)
            
            Y.append( policy_targets )
          
        # Convert the X, Y lists into numpy arrays
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        return X, Y, rewards