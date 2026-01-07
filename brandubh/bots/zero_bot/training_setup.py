#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 13:46:50 2026

@author: john

This submodule defines functions for setting up a training run for a ZeroBot.
"""

import shutil
from pathlib import Path

from .zero_network import ZeroNet, DualNet
from .brandubh_zero import ZeroBot
from .trainer import Trainer
from ..evaluate import Evaluator


NETWORKS = {
    'ZeroNet' : (ZeroNet, 'SixPlaneEncoder'),
    'DualNet' : (DualNet, 'ThreePlaneEncoder')
}


def setup_output_dir(parameter_file, params):
    """Creates the output directory for the training run creates a copy the
    parameter file used for the run.
    """
    output_dir = Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True)
    shutil.copyfile(parameter_file, output_dir / 'used_parameters.toml')
    return output_dir



def setup_bot(params):
    """Returns a ZeroBot to train.
    """
    bot_params = params['ZeroBot']
    net_params = bot_params.pop('Network', {})
    Network, encoder = NETWORKS[net_params['type']]
    net_params['encoder'] = encoder
    network = Network(net_params)
    return ZeroBot(**bot_params, network=network)



def setup_trainer(output_dir, zero_bot, params):
    """Returns a trainer object for taining the given ZeroBot.
    """
    # Create a data manager object.
    buffer_size = params['Training']['data']['buffer_size']
    epoch_size = params['Training']['data']['epoch_size']
    data_manager = zero_bot.network.get_data_manager(buffer_size, epoch_size)
    
    # Create an evaluator object.
    evaluation_rate = params['Evaluation']['evaluation_rate']
    opponents = params['Evaluation']['opponents']
    evaluator = Evaluator(output_dir, evaluation_rate, opponents)
    
    # Create and return a trainer object.
    training_params = params['Training']
    return Trainer(
        output_dir, 
        zero_bot, 
        data_manager, 
        evaluator, 
        **training_params
    )