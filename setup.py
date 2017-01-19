#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages
import brainpipe

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
        name='brainpipe',
        version=brainpipe.__version__,
        packages=find_packages(),
        description='Neural signals: data mining and machine learning',
        long_description=read('README'),
        install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'scikit-learn>=0.18',
            'joblib>=0.10.3',
        ],
        author='Etienne Combrisson',
        maintainer='Etienne Combrisson',
        author_email='e.combrisson@gmail.com',
        url='https://github.com/EtienneCmb/brainpipe',
        license=read('LICENSE'),
        include_package_data=True,
        keywords='power phase PAC feature classification machine learning neuroscience',
)
