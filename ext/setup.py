#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [
    Extension('dtwextension', sources = ['dtw.c', 'dtwextension.c'])
]

setup(
    name = 'DTWExtension',
    version = '1.0',
    include_dirs = [np.get_include()], #Add Include path of numpy
    ext_modules = ext_modules
)
