#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 15:31:19 2026

@author: brennan
"""
import os
import shutil
import unittest
import tempfile

from pathlib import Path
from brandubh.bots.zero_bot import setup_output_dir, setup_bot, setup_trainer


class TestTrainingIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Use `SHOW_TEST_OUTPUT=1 python -m unittest` to show test std output.
        cls.show_output = os.getenv('SHOW_TEST_OUTPUT', '0') == '1'
        cls.mock_parameters = {}
        cls.mock_parameters['ZeroBot'] = {
            'evals_per_turn': 70,
            'batch_size': 35,
            'c_puct': 1.4,
            'alpha': 0.15,
            'sampling_turns': 6,
            'Network': {
                'type': 'ZeroNet',
                'alpha': 0.0001,
                'backbone_depth': 7,
                'backbone_filters': 14,
                'value_filters': 14,
                'value_size_a': 14,
                'value_size_b': 7,
                'policy_filters': 14
           }
        }
        cls.mock_parameters['Training'] = {
            'num_cycles': 1,
            'episodes_per_cycle': 1,
            'move_limit': 70,
            'batch_size': 16,
            'random_move_policy': {'type': 'Constant', 'eps': 0.1},
            'data': {'buffer_size': 256, 'epoch_size': 64}
        }
        cls.mock_parameters['Evaluation'] = {
            'evaluation_rate': 7,
            'opponents': {
                'rand_1': {
                    'type': 'RandomBot',
                    'num_games': 10,
                    'turn_limit': 70,
                    'look_ahead': 1
                }
            }
        }
    
    
    @classmethod
    def tearDownClass(cls):
        pycache = Path('__pycache__')
        if pycache.exists():
            shutil.rmtree(pycache)
    
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir='.')
        self.test_output_dir = Path(self.temp_dir.name)
    
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    
    def _run(self, params, name):
        # Create mock parameter file.
        parameter_file = self.test_output_dir / 'empty_params.toml'
        with open(parameter_file, 'w') as file:
            file.writelines('test parameter file')
        
        # Start the test training run.
        output_dir = setup_output_dir(parameter_file, params)
        bot = setup_bot(params)
        trainer = setup_trainer(output_dir, bot, params)
        trainer.train()
        
        output_dir = self.test_output_dir.glob(f'{name}_test')[0]
        
        # Check if evaluations files exist.
        evaluations_path = output_dir / 'evaluations'
        is_empty = next(evaluations_path.iterdir(), None) is None
        self.assertFalse(is_empty, "No evaluations created.")
        
        # Check if model files exist.
        model_path = output_dir / 'model'
        is_empty = next(model_path.iterdir(), None) is None
        self.assertFalse(is_empty, "No model created.")
        
        # Check if experience directories exist.
        experience_path = output_dir / 'experience'
        is_empty = next(experience_path.iterdir(), None) is None
        self.assertFalse(is_empty, "No experience created.")
        
        # Check if training log file exists.
        log_path = output_dir / 'training_log.json'
        self.assertTrue(log_path.exists(), "No training log created.")
        
        
    #=========================================================================#
    #                          Integration Tests
    #=========================================================================#
    
    def test_zero_net(self):
        parameters = self.mock_parameters
        parameters['output_dir'] = 'zero_net_test'
        name = 'zero_net'
        self._run(parameters, name)
        
        
    def test_dual_net(self):
        parameters = self.mock_parameters
        parameters['output_dir'] = 'dual_net_test'
        parameters['ZeroBot']['Network']['type'] = 'DualNet'
        parameters['Training']['random_move_policy'] = {
            'type'        : 'Balanced',
            'initial_eps' : 0.0,
            'window_size' : 21,
            'gain'        : 0.1
        }
        name = 'dual_net'
        self._run(parameters, name)