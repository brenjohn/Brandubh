#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 22:13:25 2023

@author: john
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Brandubh',
    ext_modules = cythonize("brandubh/game.pyx") #, annotate=True)
)
