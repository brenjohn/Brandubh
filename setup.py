#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 22:13:25 2023

@author: john
"""

import os
import shutil

from setuptools import setup, Extension
from Cython.Build import cythonize

# setup(
#     name='Brandubh',
#     ext_modules = cythonize("brandubh/game.pyx") #, annotate=True)
# )


CYTHON_TRACE = os.getenv("CYTHON_TRACE", "0") == "1"

def clean():
    
    brandubh_dir = "brandubh"
    cache_dir    = os.path.join("brandubh", "__pycache__")
    build_dir    = "build"
    
    if os.path.exists(brandubh_dir):         
        for file in os.listdir(brandubh_dir):
            if file.endswith((".so", ".c", ".cpp")):
                os.remove(os.path.join(brandubh_dir, file))
        if os.path.exists(cache_dir):         
            shutil.rmtree(cache_dir)
    
    if os.path.exists(build_dir):         
        shutil.rmtree(build_dir)

if "clean" in os.sys.argv:
    clean()
    print("Cleaned build artifacts.")
    os.sys.exit(0)

setup(
    name='Brandubh',
    ext_modules=cythonize(
        [
            Extension(
                "brandubh.game",
                ["brandubh/game.pyx"],
                define_macros=[("CYTHON_TRACE", "1")] if CYTHON_TRACE else [],
            )
        ],
        compiler_directives={"linetrace": True}
    )
)