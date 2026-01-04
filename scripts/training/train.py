#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 15:03:50 2025

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

# configure tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

import toml
import argparse
from pathlib import Path

from brandubh.bots.zero_bot import setup_output_dir, setup_bot, setup_trainer



def main(parameter_file):
    # Read parameter file.
    with open(parameter_file, 'r') as file:
        params = toml.load(file)
    
    # Create output directory for this training run.
    output_dir = setup_output_dir(parameter_file, params)
    
    # Create the bot to be trained.
    bot = setup_bot(params)
    
    # Set up a trainer object to train and monitor the bot.
    trainer = setup_trainer(output_dir, bot, params)
    
    # Start the training process.
    trainer.train()
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a Brandubh model."
    )
    parser.add_argument(
        '--parameter_file', 
        type=Path,
        default='./train_parameters.toml',
        help="Path to the training parameter file"
    )
    
    parameter_file = parser.parse_args().parameter_file
    main(parameter_file)