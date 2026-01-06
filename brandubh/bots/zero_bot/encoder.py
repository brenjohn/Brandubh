#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 16:04:57 2026

@author: john

This submodule defines a classes used to encode game states as input tensors
that can be passed to a ZeroBot neural network. It can also decode the output 
of the network to a dictionary of move-value pairs.
"""

import numpy as np


class BaseEncoder:
    """A base class that other encoder classes should inherit from.
    """
        
    def encode(self, game_state):
        """Encode the given game state as a tensor.
        """
        pass
    
    
    def decode_policy(self, model_output, legal_moves):
        """Returns a dict of normalised priors for the given legal moves. The
        priors are taken from the given output from a policy head.
        
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
        """Encodes the given distribution over moves as a tensor similar to an
        output tensor of a policy head. Can be used to create target outputs
        for training data.
        """
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
        """Encodes the given prior distributions as an output tensor of a
        policy head.
        """          
        encoded_priors = [self.encode_prior(prior) for prior in priors]
        num_moves = len(encoded_priors)
        return np.reshape(encoded_priors, (num_moves, 7, 7, 24))
    
    
    def create_training_data(self, experience):
        """A method to convert game data in an experience list to training data
        for training a ZeroBot neural network.
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
    


class SixPlaneEncoder(BaseEncoder):
    """This class is used to encode a brandubh game state as a tensor with six
    channels (See encode method) and is intended to be used with the ZeroNet
    class.
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
        player       = game_state.player
        game_set     = game_state.game_set
        
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



class ThreePlaneEncoder(BaseEncoder):
    """This class is used to encode a brandubh game state as a tensor with 
    three channels (See encode method) and is intended to be used with the
    DualNet class.
    """
        
    def encode(self, game_state):
        """
        A game state is encoded as three 7x7 planes (or a tensor with shape 
        (7,7,3)) as described below.
        
        The first 7x7 plane is an array of 0's and 1's encoding the positions 
        of all soldier pieces owned by the white player, i.e. the array has a 
        1 in entries corresponding to squares of the board occupied by a white
        soldier.
        
        The second 7x7 plane is an array encoding the position of king piece.
        
        The third plane encodes the position of all the black pieces.
        """
        board_tensor = np.zeros((7,7,3))
        game_set = game_state.game_set
            
        # white soldier pieces
        for piece in range(3, 11, 2):
            r, c = game_set.piece_position(piece)
            board_tensor[r, c, 0] = 1
            
        # King piece
        r, c = game_set.piece_position(1)
        board_tensor[r, c, 1] = 1
        
        # black soldier pieces
        for piece in range(2, 18, 2):
            r, c = game_set.piece_position(piece)
            board_tensor[r, c, 2] = 1
        
        return board_tensor
    
    
    def create_training_data(self, experience):
        white_experience, black_experience = self.split_experience(experience)
        white_training_data = super().create_training_data(white_experience)
        black_training_data = super().create_training_data(black_experience)
        return white_training_data, black_training_data
    
    
    def split_experience(self, experience):
        white_experience, black_experience = [], []
        for episode in experience:
            white_turns, black_turns = self.split_episode(episode)
            white_experience.append(white_turns)
            black_experience.append(black_turns)
        return white_experience, black_experience
    
    
    def split_episode(self, episode):
        white_turns = [i for i, p in enumerate(episode['players']) if p == 1]
        black_turns = [i for i, p in enumerate(episode['players']) if p == -1]
        
        white_data, black_data = {}, {}
        for key, value in episode.items():
            if isinstance(value, list):
                white_data[key] = [value[i] for i in white_turns]
                black_data[key] = [value[i] for i in black_turns]
            elif isinstance(value, int):
                white_data[key] = value
                black_data[key] = value
        
        return white_data, black_data