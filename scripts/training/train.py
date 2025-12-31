#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 15:03:50 2025

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import json
import toml
import argparse
import shutil
from pathlib import Path

# configure tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

from brandubh.bots.zero_bot.brandubh_zero import ZeroBot
from brandubh.bots.zero_bot.zero_network import ZeroNet
from brandubh.bots.zero_bot.training import gain_experience, save_experience
from brandubh.bots.evaluate import Evaluator



def setup_output_dir(params):
    output_dir = Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True)
    shutil.copyfile(parameter_file, output_dir / 'used_parameters.toml')
    return output_dir



def setup_bot(params):
    bot_params = params['ZeroBot']
    net = ZeroNet()
    return ZeroBot(**bot_params, network=net)



def main(parameter_file):
    
    with open(parameter_file, 'r') as file:
        params = toml.load(file)
    
    output_dir = setup_output_dir(params)
    bot = setup_bot(params)
    
    train_params = params['Training']
    num_episodes = train_params['num_episodes']
    move_limit = train_params['move_limit']
    train_size = train_params['train_size']
    batch_size = train_params['batch_size']
    num_cycles = train_params['num_cycles'] # TODO: use this
    eps = 0.07
    
    evaluation_rate = params['evaluation_rate']
    evaluation_opponents = params['Evaluation']
    evaluator = Evaluator(output_dir, evaluation_opponents)
    
    cycle = -1
    data_manager = bot.get_DataManager()
    while True:
        cycle += 1
        
        print('\nGainning experience, cycle {0}'.format(cycle))
        exp = gain_experience(bot, num_episodes, move_limit, eps)
        save_experience(output_dir, cycle, exp)
        
        print('Preparing training data')
        # Add the generated experience to the bank of training data and load 
        # all training data
        training_data = bot.network.create_training_data(exp)
        data_manager.append_data(training_data)
        
        print('\nTraining network, cycle {0}'.format(cycle))
        training_data = data_manager.sample_training_data(train_size)
        bot.network.train(training_data, batch_size=batch_size)
        bot.save_bot("model_data/model_curr_data/")
        
        if cycle % evaluation_rate == 0:
            evaluator.evaluate(bot)
        


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