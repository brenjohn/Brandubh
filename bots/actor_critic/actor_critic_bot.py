#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:31:20 2020

@author: john
"""

import numpy as np

from brandubh import Act, GameState
from random_bot import RandomBot
from four_plane_encoder import FourPlaneEncoder
from keras.models import Model, load_model
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.optimizers import SGD

class ActorCriticBot():
    
    def __init__(self, model=None):
        
        self.encoder = FourPlaneEncoder()
        self.evaluation_history_old = []
        self.evaluation_history_ran = []
        self.rand_bot = RandomBot()
        if model:
            self.model = model
        else:
            self.model = self.init_model()
            
            
            
    def init_model(cls):
        
        # Create the network
        board_input = Input(shape=(7,7,4), name='board_input')
        
        # conv1 = Conv2D(64, (3, 3), 
        #                padding='same',
        #                activation='sigmoid')(board_input)
        
        # conv2 = Conv2D(64, (3, 3), 
        #                padding='same',
        #                activation='sigmoid')(conv1)
        
        flat = Flatten()(board_input)
        hidden_board1 = Dense(512, activation='sigmoid')(flat)
        hidden_board2 = Dense(512, activation='sigmoid')(hidden_board1)
        processed_board = Dense(512, activation='sigmoid')(hidden_board2)
        
        policy_hidden = Dense(512, activation='sigmoid')(processed_board)
        policy_output = Dense(96, activation='softmax')(policy_hidden)
        
        value_hidden = Dense(512, activation='sigmoid')(processed_board)
        value_output = Dense(1, activation='tanh')(value_hidden)
        
        model = Model(inputs=board_input, 
                      outputs=[policy_output, value_output])
        
        
        # Complie model
        model.compile(optimizer=SGD(lr=0.001),
                      loss=['categorical_crossentropy', 'mse'],
                      loss_weights=[1.0, 1.0])
        
        return model
        
    
        
    def select_move(self, game_state):
        
        move_probs, move_val, input_tensor = self.predict(game_state)
        
        move_probs = move_probs**3
        move_probs /= np.sum(move_probs)
        
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1-eps)
        move_probs /= np.sum(move_probs)
        
        
        candidates = np.arange(96)
        ranked_moves = np.random.choice(candidates, 96,
                                        replace=False,
                                        p=move_probs)
        
        pieces = np.sum(input_tensor[:,:,:,:2],-1).reshape(7,7)
        
        for move_idx in ranked_moves:
            move = self.encoder.decode_move_index(pieces, move_idx)
            if not game_state.is_move_illegal(move):
                    return Act.play(move), move_val
                
        return Act.pass_turn(), None
    
    
    
    def predict(self, game_state):
        input_tensor = self.encoder.encode(game_state).reshape(1,7,7,4)
        probs, value = self.model.predict(input_tensor)
        return probs[0], value[0][0], input_tensor
    
    
    
    def evaluate_against_rand_bot(self, num_games):
        act_crit_player = 1
        score = 0
        num_games_won_as_black = 0
        num_games_won_as_white = 0
        
        for i in range(num_games):
            print('\rEvaluating against rand bot: game {0}'.format(i),end='')
            game = GameState.new_game()
            
            max_num_of_turns = 1000
            turns_taken = 0
            
            while game.is_not_over() and turns_taken < max_num_of_turns:
                if game.player == act_crit_player:
                    action, value = self.select_move(game)
                else:
                    action = self.rand_bot.select_move(game)
                    
                game.take_turn_with_no_checks(action)
                turns_taken += 1
             
                
            if turns_taken < max_num_of_turns:
                score += act_crit_player*game.winner
            
                if act_crit_player == game.winner:
                    if act_crit_player == 1:
                        num_games_won_as_white += 1
                    else:
                        num_games_won_as_black += 1
                        
                act_crit_player *= -1
                        
            else:
                score -= 1
                act_crit_player *= -1
        
        # Save the evaluation score of the bot along with fraction of games
        # won as black/white and the total number of games
        self.evaluation_history_ran.append([score/num_games,
                                            2*num_games_won_as_white/num_games,
                                            2*num_games_won_as_black/num_games,
                                            num_games] )
        
        
        
    def evaluate_against_old_bot(self, num_games):
        
        model = load_model('old_actor_critic_model.h5')
        old_bot = ActorCriticBot(model)
        
        act_crit_player = 1
        score = 0
        num_games_won_as_black = 0
        num_games_won_as_white = 0
        
        for i in range(num_games):
            print('\rEvaluating against old bot: game {0}'.format(i),end='')
            game = GameState.new_game()
            
            max_num_of_turns = 1000
            turns_taken = 0
            
            while game.is_not_over() and turns_taken < max_num_of_turns:
                if game.player == act_crit_player:
                    action, value = self.select_move(game)
                else:
                    action, value = old_bot.select_move(game)
                    
                game.take_turn_with_no_checks(action)
                turns_taken += 1
                  
                
            if turns_taken < max_num_of_turns:
                score += act_crit_player*game.winner
            
                if act_crit_player == game.winner:
                    if act_crit_player == 1:
                        num_games_won_as_white += 1
                    else:
                        num_games_won_as_black += 1
                        
                act_crit_player *= -1
                        
            else:
                score -= 1
                act_crit_player *= -1
            
        self.evaluation_history_old.append([score/num_games,
                                            2*num_games_won_as_white/num_games,
                                            2*num_games_won_as_black/num_games,
                                            num_games] )
    
    
    
    def save_bot(self):
        self.model.save('actor_critic_model.h5')
        np.save('eval_history_old.npy', self.evaluation_history_old)
        np.save('eval_history_ran.npy', self.evaluation_history_ran)
        
    def load_bot(self):
        self.model = load_model('actor_critic_model.h5')
        self.evaluation_history_old = list( np.load('eval_history_old.npy') )
        self.evaluation_history_ran = list( np.load('eval_history_ran.npy') )
        
    def save_as_old_bot(self):
        self.model.save('old_actor_critic_model.h5')