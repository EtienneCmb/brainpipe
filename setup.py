#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='brainpipe',
    version='0.1.8',
    packages=find_packages(),
    description='Bridge between neural signal datasets and machine learning',
    long_description=read('README.md'),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'joblib',
        'matplotlib',
        'psutil',
    ],
    dependency_links=[],
    author='Etienne Combrisson',
    maintainer='Etienne Combrisson',
    author_email='e.combrisson@gmail.com',
    url='https://github.com/EtienneCmb/brainpipe',
    license=read('LICENSE'),
    include_package_data=True,
    keywords='brain neuroscience features classification machine learning phase-amplitude coupling',
    classifiers=["Development Status :: 3 - Alpha",
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Education',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Visualization',
                 "Programming Language :: Python :: 3.5"
                 ])
