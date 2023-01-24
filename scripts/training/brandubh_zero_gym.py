#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:19:04 2022

@author: john

This script will spawn several processes: one to train a ZeroBot with self play
experience and others to evaluate how good the bot is against various opponents
at different stages during training. Evaluation results are written to separate
files for different opponents. 
"""
import os
import sys
sys.path.append("..")
sys.path.append("../..")

import numpy as np
import logging
import multiprocessing
from multiprocessing import Process, Manager

from brandubh.bots.zero_bot.brandubh_zero import ZeroBot
from brandubh.bots.zero_bot.networks.zero_network import ZeroNet
from brandubh.bots.zero_bot.networks.dual_network import DualNet
from brandubh.bots.training_utils import gain_experience
from brandubh.bots.training_utils import save_training_data



#%%
def write_results(filename, results):
    file = open(filename, "a")
    l = ""
    for r in results:
        l += str(r) + " "
    file.write(l + "\n")
    file.close()
    

def rand_evaluation_task(lock, out_dir):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fname = out_dir + 'eval_process_rand.log'
    logging.basicConfig(filename=fname, level=logging.DEBUG)

    logger.info("creating")
    # net = ZeroNet()
    net = DualNet()
    bot = ZeroBot(evals_per_turn=700, batch_size=35, network=net)
    bot.compile_network((1.0, 0.1))
    
    logger.info("loading")
    with lock:
        bot.load_bot("model_data/model_curr_data/")
    
    logger.info("evaluating")
    games_to_play = 210
    while True:
        moves_to_look_ahead = 1
        bot.evaluate_against_rand_bot(games_to_play, moves_to_look_ahead)
        fname = out_dir + "rand_lookahead_{0}.txt".format(moves_to_look_ahead)
        write_results(fname, bot.evaluation_history_rand[-1])
        
        moves_to_look_ahead = 140
        bot.evaluate_against_rand_bot(games_to_play, moves_to_look_ahead)
        fname = out_dir + "rand_lookahead_{0}.txt".format(moves_to_look_ahead)
        write_results(fname, bot.evaluation_history_rand[-1])
        
        moves_to_look_ahead = 1
        bot.evaluate_against_grnd_bot(games_to_play, moves_to_look_ahead)
        fname = out_dir + "grnd_lookahead_{0}.txt".format(moves_to_look_ahead)
        write_results(fname, bot.evaluation_history_grnd[-1])
        
        moves_to_look_ahead = 140
        bot.evaluate_against_grnd_bot(games_to_play, moves_to_look_ahead)
        fname = out_dir + "grnd_lookahead_{0}.txt".format(moves_to_look_ahead)
        write_results(fname, bot.evaluation_history_grnd[-1])
        
        with lock:
            bot.load_bot("model_data/model_curr_data/")
   
        
   
def mcts_evaluation_task(lock, look_ahead, out_dir):    
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fname = out_dir + 'eval_proc_mcts_{0}.log'.format(look_ahead)
    logging.basicConfig(filename=fname, level=logging.DEBUG)
    
    logger.info("creating")
    # net = ZeroNet()
    net = DualNet()
    bot = ZeroBot(evals_per_turn=700, batch_size=35, network=net)
    bot.compile_network((1.0, 0.1))
    
    logger.info("loading")
    with lock:
        bot.load_bot("model_data/model_curr_data/")
    
    logger.info("evaluating")
    games_to_play = 100
    turn_limit = 350
    while True:
        bot.evaluate_against_mcts_bot(games_to_play, 
                                      look_ahead, 
                                      turn_limit, 
                                      logger)
        fname = out_dir + "mcts_with_lookahead_{0}.txt".format(look_ahead)
        write_results(fname, bot.evaluation_history_mcts[-1])
            
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
        
        # If the directory for the evaluation logs doesn't exist, create it.
        out_dir = 'evaluation_logs/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists("model_data/model_curr_data/"):
            os.makedirs("model_data/model_curr_data/")
        
        # Initialise bot and save current weights.
        # net = ZeroNet()
        net = DualNet()
        bot = ZeroBot(evals_per_turn=2100, batch_size=35, network=net)
        bot.compile_network((1.0, 0.1))
        
        with lock:
            bot.save_bot("model_data/model_curr_data/")
        
        # Spawn evaluator processes.
        rand_eval_process = Process(target=rand_evaluation_task, 
                                    args=(lock, out_dir))
        rand_eval_process.start()
        
        look_ahead = 350
        mcts_eval_process = Process(target=mcts_evaluation_task, 
                                    args=(lock, look_ahead, out_dir))
        mcts_eval_process.start()
        
        # Start training the bot with self-play games.
        num_episodes = 8
        move_limit = 140
        dm = bot.get_DataManager()
        cycle = 0
        while True:
            cycle += 1
            print('\nGainning experience, cycle {0}'.format(cycle))
            eps = 0.07
            exp = gain_experience(bot, bot, num_episodes, move_limit, eps)
            
            print('Preparing training data')
            # Add the generated experience to the bank of training data and
            # load all training data
            training_data = bot.network.create_training_data(exp)
            save_training_data(training_data, cycle)
            dm.append_data(training_data)
            
            print('\nTraining network, cycle {0}'.format(cycle))
            n = cycle if cycle < 7 else 7
            for i in range(n):
                training_data = dm.sample_training_data(4096)
                losses = bot.network.train(training_data, batch_size=256)
            
            bot.compile_network((1.0, 0.1))
            with lock:
                bot.save_bot("model_data/model_curr_data/")
            dm.save("model_data/model_curr_data/appended.npy")