#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 22:40:16 2019

@author: john
"""

import argparse
import brandubh
import numpy as np

from bots.mcbot import MCTSBot
from bots.four_plane_encoder import FourPlaneEncoder
from usr_v_bot import print_board


def generate_game(rounds, max_moves, temperature):
    
    boards, moves = [], []
    
    encoder = FourPlaneEncoder()
    
    game = brandubh.GameState.new_game()
    
    bot = MCTSBot(rounds, temperature)
    
    num_moves = 0
    
    while game.is_not_over():
        # print_board(game, 'temp')
        print('\rCalculating move {0}'.format(num_moves+1), end='')
        move = bot.select_move(game)
        if move.is_play:
            board_tensor = encoder.encode(game)
            boards.append(board_tensor)
            
            pieces = np.sum(board_tensor[:,:,:2],-1).reshape((7,7))
            move_one_hot = np.zeros(96)
            move_one_hot[encoder.encode_move(pieces, move.move)] = 1
            moves.append(move_one_hot)
            
        game.take_turn(move)
        num_moves += 1
        if num_moves > max_moves:
            break
        
    return np.array(boards), np.array(moves), game.winner


parser = argparse.ArgumentParser()
parser.add_argument('--rounds', '-r', type=int, default=20000)
parser.add_argument('--temperature', '-t', type=float, default=0.8)
parser.add_argument('--max-moves', '-m', type=int, default=100,
                    help='Max moves per game')
parser.add_argument('--num-games', '-n', type=int, default=60)
parser.add_argument('--board-out', default='data_60.npy')
parser.add_argument('--move-out', default='labels_60.npy')

args = parser.parse_args()
xs = []
ys = []
ws = []

for i in range(args.num_games):
    print('Generating game %d/%d...' % (i+1, args.num_games))
    x, y, w = generate_game(args.rounds, args.max_moves, args.temperature)
    xs.append(x)
    ys.append(y)
    ws.append(w)
    
x = np.concatenate(xs)
y = np.concatenate(ys)

np.save(args.board_out, x)
np.save(args.move_out, y)
