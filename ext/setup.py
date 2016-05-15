#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [
    Extension('pairwisedist', sources = ['dtw.c', 'pairwisedist.c'])
]

setup(
    name = 'PairWiseDist',
    version = '1.0',
    include_dirs = [np.get_include()], #Add Include path of numpy
    ext_modules = ext_modules
)
