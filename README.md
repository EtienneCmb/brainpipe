# brainpipe

![alt tag](https://travis-ci.org/EtienneCmb/brainpipe.svg?branch=master)

Version : 0.1.0

## Description

Brainpipe is a toolbox dedicated to the analysis and classification of neuro-physiological signals. Brainpipe allows for the extraction of a wide range of features from brain signals and the application of a wide range of machine learning algorithms based on scikit-learn. For now the available processing and feature extraction functionlities are tailored to EEG, MEG, intracranial EEG (SEEG & ECoG) signals. Brainpipe can be used for single and multi-feature classification and provides a flexible and easiliy extendable tool for neuroscience data mining. Checkout the [documentation](https://etiennecmb.github.io/) and see intructions for installation.

Brainpipe is a package primarily written by Etienne Combrisson, as part of his PhD research work, suprvised by Prof Karim Jerbi (CoCo Lab, Psychology Department, University of Montreal, QC, Canada) and Prof Aymeric Guillot (University Claude Bernard Lyon 1, Inter-University Laboratory of Human Movement Biology, Lyon, France).

The current version of Brainpipe consists of the following modules:
- bpstudy : managed features / file database
- Extracts anatomical information using mni/talairach coordinates of sites (e.g. electrode locations in intracranial EEG)
- feature : extracts a wide range of features, including: power, phase, phase-amplitude coupling
- Optimized classification and feature combination using a range of techniques based on scikit-learn 

![brainpipe](https://github.com/EtienneCmb/brainpipe/blob/master/docs/image/bplogo.png "brainpipe")

## Modules

### bpstudy
Create and manage many studies, without carrying of path, variable names or settings. Everything is going to be self organized in the clearest way as possible.

### physiology
This module give the physiological informations of intracranial recordings. It will return the brodmann area, the gyrus, the lobe and the hemisphere for a given list of electrodes' coordonates.
The results should be the same as Talairach Daemon

### feature
Extract time resolved features from original signals. Here's the current list of extractable features:
- Filtered signal (using butterworth / bessel / eegfilt filters)
- Power (hilbert / wavelet)
- Phase (based on hilbert transform)
- Phase-amplitude coupling (10 methods) with a variety of normalizations and surrogates computing methods
- Entropy (coming soon)
- Kurtosis (coming soon)
- Fractales, C1 // C2 (very futur)

### classification
Classify each time resolved features using parallel computing. There is also several methods to evaluate the statistical significiance of decoding accuracies:
- binomial
- permutations
	- shuffle labels
	- full randomization
	- intra-class shuffling

The classification modules provide the basics classifiers and cross-validations implemented in scikit-learn with an optimization for large array (which is convenient for features classification).
The classification module also include:
- Time generalization : generalize the decoding performance of features across time

### multifeatures
This module include several well known methods for computing multi-features. It includes :
- Select all the features
- forward / backward / exhaustif features selection
- statistical selection (same methods as in classification module)
- nbest : select nbest features
All this methods can be used individually or they can be combined in a sequential way. An other interesting feature, is that for a given set of features, you can define groups and do a multi-features inside each group.

## Installation

## Version
v0.0 compatible with python 3.x only
(Still in development, final codes versions in xPOO for instance)

## Keywords
stereotactic electroencephalography, sEEG, iEEG, intracranial, micro-electrodes, ecog, power, phase-amplitude coupling, pac, phase, entropy, permutations, classification, brodmann, python, time generalization
