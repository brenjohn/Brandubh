#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 14:03:30 2026

@author: brennan
"""

import os

def setup_tensorflow():
    """Sets environment variables to configure tensorflow.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    
    # Configure GPU memory growth.
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices:
        for device in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError as e:
                print(f"TensorFlow Initializasion error: {e}")

# Run this on import.
setup_tensorflow()