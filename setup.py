#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='brainpipe',
    version='0.2.0',
    description='Neural signals: data mining and machine learning',
    long_description=readme,
    author='Etienne Combrisson',
    author_email='e.combrisson@gmail.com',
    url='https://github.com/EtienneCmb/brainpipe',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)

