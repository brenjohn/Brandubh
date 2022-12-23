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
# from networks.zero_network import print_tensor

from brandubh_zero import ZeroBot
from greedy_random_bot import GreedyRandomBot
from training_utils import gain_experience, save_training_data, DataManager, competition

# from multiprocessing import Process, Lock
from threading import Thread, Lock
import os

from networks.zero_network import ZeroNet

os.system("taskset -p 0xff %d" % os.getpid())


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
    
def evaluation_task(bot, lock):
    print("loading")
    lock.acquire()
    bot.load_bot("model_data/model_curr_data/")
    lock.release()
    
    print("evaluating")
    count = 0
    while True:
        moves_to_look_ahead = 1
        bot.evaluate_against_rand_bot(7, moves_to_look_ahead)
        write_results("test.txt", bot.evaluation_history_rand[-1])
        
        # if count % 7 == 0:
        #     bot.evaluate_against_mcts_bot()
        #     count = 0
            
        lock.acquire()
        bot.load_bot("model_data/model_curr_data/")
        lock.release()
        count += 1


#%%
if __name__ == '__main__':
    lock = Lock()
    
    # Initialise bot and save current weights.
    net = ZeroNet()
    bot = ZeroBot(evals_per_turn=70, batch_size=28, network=net)
    compile_bot(bot)
    
    eval_net = ZeroNet()
    eval_bot = ZeroBot(evals_per_turn=70, batch_size=28, network=eval_net)
    compile_bot(eval_bot)
    
    lock.acquire()
    bot.save_bot("model_data/model_curr_data/")
    lock.release()
    
    # Spawn evaluator process
    evaluation_process = Thread(target=evaluation_task, args=(eval_bot, lock))
    evaluation_process.start()
    # evaluation_process.join()
    
    # Start training
    num_episodes = 4
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
        lock.acquire()
        bot.save_bot("model_data/model_curr_data/")
        lock.release()