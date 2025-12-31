#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 20:31:28 2025

@author: john
"""

import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_brandubh(output_dir, board_tensor, turn_number):
    # Convert input list to numpy array if necessary
    board = np.array(board_tensor)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 1. Create the checkered background
    # A 7x7 grid of alternating 0s and 1s
    checkers = np.indices((7, 7)).sum(axis=0) % 2
    ax.imshow(checkers, cmap='Greys', alpha=0.15, origin='upper')

    # 2. Iterate through the grid to place pieces
    white = (board[0, 0, 4] == 1)
    king_plane, white_plane, black_plane = (1, 0, 2) if white else (3, 2, 0)
    for r in range(7):
        for c in range(7):
            # Check piece layers
            if board[r, c, king_plane] == 1.0:    # King
                ax.text(
                    c, r, '♔', ha='center', va='center', 
                    fontsize=40, color='black', weight='bold'
                )
            elif board[r, c, white_plane] == 1.0:  # Defender
                ax.text(
                    c, r, '♙', ha='center', va='center', 
                    fontsize=40, color='black'
                )
            elif board[r, c, black_plane] == 1.0:  # Attacker
                ax.text(
                    c, r, '♟', ha='center', va='center', 
                    fontsize=40, color='black'
                )

    # 3. Formatting the grid
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(7))
    ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    ax.set_yticklabels(range(1, 8))
    
    # Draw grid lines between squares
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Highlight the Throne (center) and Escapes (corners)
    special_squares = [(3, 3), (0, 0), (0, 6), (6, 0), (6, 6)]
    for sr, sc in special_squares:
        rect = plt.Rectangle(
            (sc-0.5, sr-0.5), 1, 1, fill=True, 
            color='black', linewidth=1, alpha=0.5
        )
        ax.add_patch(rect)

    plt.title(f"Turn {turn_number}", fontsize=16)
    
    filename = output_dir / f'board_turn_{turn_number}.png'
    plt.savefig(filename, dpi=140)
    plt.close()


def main(game):
    output_dir = Path('./game_plots/')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for turn, board in enumerate(game['boards']):
        plot_brandubh(output_dir, board, turn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Path to game JSON file."
    )
    parser.add_argument(
        '--game_file', 
        type=Path,
        default='./train_run_1/cycle_0_experience/game_2.json',
        help="Path to the training parameter file"
    )
    
    game_file = parser.parse_args().game_file
    with open(game_file, 'r') as file:
        game = json.load(file)
        
    main(game)