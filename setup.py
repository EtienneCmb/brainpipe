#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
        name='brainpipe',
        version='0.3.0',
        packages=['brainpipe'],
        description='Neural signals: data mining and machine learning',
        long_description=readme,
        install_requires=[
            'numpy',
            'scipy',
            'pandas',
            'matplotlib',
            'scikit-learn',
            'joblib',
        ],
        author='Etienne Combrisson',
        maintainer='Etienne Combrisson',
        author_email='e.combrisson@gmail.com',
        url='https://github.com/EtienneCmb/brainpipe',
        license=license,
        keywords='power phase PAC feature classification machine learning neuroscience',
)
