#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:41:39 2026

@author: brennan
"""

import brandubh.init_tf
import unittest
import sys

def run_brandubh_tests():
    loader = unittest.TestLoader()
    suite = loader.discover('test', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=1, buffer=True)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    run_brandubh_tests()