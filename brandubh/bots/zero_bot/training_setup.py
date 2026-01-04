#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 13:46:50 2026

@author: john
"""

import shutil
from pathlib import Path

from .zero_network import ZeroNet
from .brandubh_zero import ZeroBot
from .trainer import Trainer
from .data_manager import DataManager
from ..evaluate import Evaluator



def setup_output_dir(parameter_file, params):
    output_dir = Path(params['output_dir'])
    output_dir.mkdir(exist_ok=True)
    shutil.copyfile(parameter_file, output_dir / 'used_parameters.toml')
    return output_dir



def setup_bot(params):
    bot_params = params['ZeroBot']
    net = ZeroNet()
    return ZeroBot(**bot_params, network=net)



def setup_trainer(output_dir, zero_bot, params):
    buffer_size = params['Training']['data']['buffer_size']
    epoch_size = params['Training']['data']['epoch_size']
    data_manager = DataManager(buffer_size, epoch_size)
    
    evaluation_rate = params['Evaluation']['evaluation_rate']
    opponents = params['Evaluation']['opponents']
    evaluator = Evaluator(output_dir, evaluation_rate, opponents)
    
    training_params = params['Training']
    
    return Trainer(
        output_dir, 
        zero_bot, 
        data_manager, 
        evaluator, 
        **training_params
    )