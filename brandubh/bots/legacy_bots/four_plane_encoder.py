#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:37:32 2019

@author: john
"""

import numpy as np
import copy


class FourPlaneEncoder:
    
    def __init__(self):
        self.convert_to_tensor_element = {1: 'board_tensor[r,c,0] = 1',
                                          2: 'board_tensor[r,c,0] = 1;' +
                                             'board_tensor[r,c,1] = 1',
                                         -1: 'board_tensor[r,c,2] = 1',
                                         -2: 'board_tensor[r,c,2] = 1;' +
                                             'board_tensor[r,c,3] = 1'}
    
    
    def encode(self, game_state):
        board_tensor = np.zeros((7,7,4))
        player = game_state.player
        
        for (r, c) in game_state.game_set.white_pieces:
            piece = game_state.game_set.board[(r, c)]
            exec(self.convert_to_tensor_element[piece*player])
            
        for (r, c) in game_state.game_set.black_pieces:
            # piece = game_state.game_set.board[(r, c)]
            exec(self.convert_to_tensor_element[-1*player])
        
        return board_tensor
    
    
    def encode_move(self, pieces, move):
        ri, ci, rf, cf = move
        
        if ci == cf:
            move_num = rf if rf < ri else rf-1
        elif ri == rf:
            move_num = cf+6 if cf < ci else cf+6-1
        else:
            return None
        
        piece_num = 0
        for r in range(7):
            for c in range(7):
                if c == ci and r == ri:
                    return piece_num*12 + move_num
                elif pieces[r,c] == 1:
                    piece_num += 1
        return None
    
    def decode_move_index(self, game_state, index):
        piece_num = index // 12
        move_num = index % 12
        
        if game_state.player == 1:
            pieces = game_state.game_set.white_pieces
        else:
            pieces = game_state.game_set.black_pieces
            
        if piece_num >= len(pieces):
            return (0,0,0,0)
            
        pieces = sorted(pieces)
        
        ri, ci = pieces[piece_num]
        if move_num < 6:
            cf = ci
            rf = move_num if move_num<ri else move_num+1
        else:
            rf = ri
            move_num -= 6
            cf = move_num if move_num<ci else move_num+1
        return (ri, ci, rf, cf)
    
    
    def decode_move_index_from_pieces(self, pieces, index):
        piece_num = index // 12
        move_num = index % 12
        
        piece = 0
        for r in range(7):
            for c in range(7):
                if pieces[r,c]==1:
                    if piece == piece_num:
                        ri, ci = r, c
                        if move_num < 6:
                            cf = ci
                            rf = move_num if move_num<ri else move_num+1
                        else:
                            rf = ri
                            move_num -= 6
                            cf = move_num if move_num<ci else move_num+1
                        return (ri, ci, rf, cf)
                    piece += 1
        return (0,0,0,0)
    
    
    def shape(self):
        return 7, 7, 4
    
    
    def expand_data(self, X, Y):
        
        move_val_maps, pieces = [], []
        for n,y in enumerate(Y):
            pieces.append(X[n,:,:,0].reshape((7,7)))
            move_val_map = {self.decode_move_index_from_pieces(pieces[n], ind) : val
                            for ind, val in enumerate(y)}
            move_val_maps.append(move_val_map)
        
        X_temp = copy.copy(X)
        
        X_temp = np.transpose(X_temp, (0,2,1,3))
        Y_temp, move_val_maps, pieces = self.transpose_policy(move_val_maps, 
                                                              pieces)
        
        X = np.concatenate((X, copy.copy(X_temp)), axis=0)
        Y = np.concatenate((Y, Y_temp), axis=0)
        
        for i in range(3):
            
            X_temp = np.flip(X_temp, axis=1)
            Y_temp, move_val_maps, pieces = self.flip_policy(move_val_maps, 
                                                             pieces)
            
            X = np.concatenate((X, copy.copy(X_temp)), axis=0)
            Y = np.concatenate((Y, Y_temp), axis=0)
            
            X_temp = np.transpose(X_temp, (0,2,1,3))
            Y_temp, move_val_maps, pieces = self.transpose_policy(move_val_maps, 
                                                              pieces)
            
            X = np.concatenate((X, copy.copy(X_temp)), axis=0)
            Y = np.concatenate((Y, Y_temp), axis=0)
            
        return X, Y
    
    def transpose_policy(self, move_val_maps, boards):
        n = len(move_val_maps)
        Y = np.zeros((n, 96))
        
        for i in range(n):
            move_val_map = move_val_maps[i]
            pieces = boards[i]
            
            piece = 0
            for c in range(7):
                for r in range(7):
                    
                    if pieces[r,c] == 1:
                        row_moves = [*range(r), *range(r+1, 7)]
                        for move_num, ri in enumerate(row_moves):
                            ind = piece*12 + move_num + 6
                            Y[i, ind] = move_val_map[(r,c,ri,c)]
                            
                        col_moves = [*range(c), *range(c+1, 7)]
                        for move_num, ci in enumerate(col_moves):
                            ind = piece*12 + move_num
                            Y[i, ind] = move_val_map[(r,c,r,ci)]
                        piece += 1
                        
            transposed_move_val_map = {(ci,ri,cf,rf) : val
                                       for (ri,ci,rf,cf), val 
                                       in move_val_map.items()}
            move_val_maps[i] = transposed_move_val_map
            boards[i] = np.transpose(pieces)
                        
        return Y, move_val_maps, boards
    
    
    def flip_policy(self, move_val_maps, boards):
        n = len(move_val_maps)
        Y = np.zeros((n, 96))
        
        for i in range(n):
            move_val_map = move_val_maps[i]
            pieces = boards[i]
            
            piece = 0
            for r in range(6,-1,-1):
                for c in range(7):
                    
                    if pieces[r,c] == 1:
                        row_moves = np.flip([*range(r), *range(r+1, 7)])
                        for move_num, ri in enumerate(row_moves):
                            ind = piece*12 + move_num
                            Y[i, ind] = move_val_map[(r,c,ri,c)]
                            
                        col_moves = [*range(c), *range(c+1, 7)]
                        for move_num, ci in enumerate(col_moves):
                            ind = piece*12 + move_num + 6
                            Y[i, ind] = move_val_map[(r,c,r,ci)]
                        piece += 1
                        
            flipped_move_val_map = {(6-ri,ci,6-rf,cf) : val
                                     for (ri,ci,rf,cf), val 
                                     in move_val_map.items()}
            move_val_maps[i] = flipped_move_val_map
            boards[i] = np.flip(pieces, axis=0)
                        
        return Y, move_val_maps, boards
        