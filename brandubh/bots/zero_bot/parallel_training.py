#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:19:04 2022

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
import logging
import multiprocessing
from multiprocessing import Process, Manager

from networks.zero_network import ZeroNet
from brandubh_zero import ZeroBot
from training_utils import gain_experience, save_training_data, DataManager


def compile_bot(bot):
    bot.network.model.compile(optimizer=keras.optimizers.Adam(),
                              loss=['categorical_crossentropy', 'mse'],
                              loss_weights=[1.0, 0.1])



#%%
def write_results(filename, results):
    file = open(filename, "a")
    l = ""
    for r in results:
        l += str(r) + " "
    file.write(l + "\n")
    file.close()
    
    
    
def rand_evaluation_task(lock):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename='eval_process_rand.log', level=logging.DEBUG)

    logger.info("creating")
    net = ZeroNet()
    bot = ZeroBot(evals_per_turn=700, batch_size=35, network=net)
    compile_bot(bot)
    
    logger.info("loading")
    with lock:
        logger.info("lock acquired")
        bot.load_bot("model_data/model_curr_data/")
    
    logger.info("evaluating")
    games_to_play = 210
    while True:
        moves_to_look_ahead = 1
        bot.evaluate_against_rand_bot(games_to_play, moves_to_look_ahead)
        write_results("rand_no_lookahead.txt", bot.evaluation_history_rand[-1])
        
        moves_to_look_ahead = 140
        bot.evaluate_against_rand_bot(games_to_play, moves_to_look_ahead)
        write_results("rand_with_lookahead.txt", bot.evaluation_history_rand[-1])
        
        moves_to_look_ahead = 1
        bot.evaluate_against_grnd_bot(games_to_play, moves_to_look_ahead)
        write_results("grnd_no_lookahead.txt", bot.evaluation_history_grnd[-1])
        
        moves_to_look_ahead = 140
        bot.evaluate_against_grnd_bot(games_to_play, moves_to_look_ahead)
        write_results("grnd_with_lookahead.txt", bot.evaluation_history_grnd[-1])
        
        with lock:
            bot.load_bot("model_data/model_curr_data/")
   
        
   
def mcts_evaluation_task(lock):    
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(filename='eval_process_mcts.log', level=logging.DEBUG)
    
    logger.info("creating")
    net = ZeroNet()
    bot = ZeroBot(evals_per_turn=700, batch_size=35, network=net)
    compile_bot(bot)
    
    logger.info("loading")
    with lock:
        logger.info("lock acquired")
        bot.load_bot("model_data/model_curr_data/")
    
    logger.info("evaluating")
    games_to_play = 100
    moves_to_look_ahead = 350
    while True:
        bot.evaluate_against_mcts_bot(games_to_play, moves_to_look_ahead)
        write_results("mcts_with_lookahead.txt", bot.evaluation_history_mcts[-1])
            
        with lock:
            bot.load_bot("model_data/model_curr_data/")


#%%
if __name__ == '__main__': 
    with Manager() as manager:
        lock = manager.Lock()
        multiprocessing.log_to_stderr()
        multiprocessing.set_start_method('spawn', force=True)
        logger = multiprocessing.get_logger()
        logger.setLevel(logging.INFO)
        
        # Initialise bot and save current weights.
        net = ZeroNet()
        bot = ZeroBot(evals_per_turn=2100, batch_size=35, network=net)
        compile_bot(bot)
        
        with lock:
            bot.save_bot("model_data/model_curr_data/")
        
        # Spawn evaluator process
        rand_eval_process = Process(target=rand_evaluation_task, args=(lock,))
        rand_eval_process.start()
        # rand_eval_process.join()
        mcts_eval_process = Process(target=mcts_evaluation_task, args=(lock,))
        mcts_eval_process.start()
        # mcts_eval_process.join()
        
        # Start training
        num_episodes = 8
        num_cycles = 1
        move_limit = 140
        
        dm = DataManager()
        cycle = 0
        while True:
            cycle += 1
            print('\nGainning experience, cycle {0}'.format(cycle))
            eps = 0.07
            experience = gain_experience(bot, bot, num_episodes, move_limit, eps)
            
            print('Preparing training data')
            # Add the generated experience to the bank of training data and load
            # all training data
            training_data = bot.network.create_training_data(experience)
            save_training_data(training_data, cycle)
            dm.append_data(training_data)
            
            print('\nTraining network, cycle {0}'.format(cycle))
            n = cycle + 1 if cycle < 7 else 7
            for i in range(n):
                training_data = dm.sample_training_data(4096)
                losses = bot.network.train(training_data, batch_size=256)
            
            compile_bot(bot)
            with lock:
                bot.save_bot("model_data/model_curr_data/")
            np.save("model_data/model_curr_data/appended.npy", dm.appended)
