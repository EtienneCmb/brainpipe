.. _refpart:

References
==========

This is a non exhaustive list of function and description of what is actually implemented in brainpipe. This list correspond to the most usefull functions.


Pre-processing
--------------
Bundle of functions to pre-process data.

.. code-block:: python

    from brainpipe.preprocessing import *

.. cssclass:: table-striped

==============          ==================================================================================
Function                Description
==============          ==================================================================================
bipolarization          Bipolarise stereotactic eeg electrodes
xyz2phy                 Get physiological informations about structures using MNI or Talairach coordinates
==============          ==================================================================================


Features
--------
Bundle of functions to extract features from neural signals.

.. code-block:: python

    from brainpipe.features import *

.. cssclass:: table-striped

================        ==================================================================================
Function                Description
================        ==================================================================================
sigfilt                 Filtered signal only
amplitude               Amplitude of the signal
power                   Power of the signal
phase                   Phase of the signal
PLF                     Phase-Locking Factor
TF                      Time-frequency maps
pac                     Phase-Amplitude Coupling (large variety of methods)
PhaseLockedPower        Time-frequency maps phase locked to a specific phase
erpac                   Event Related Phase-Amplitude Coupling (time-resolved pac)
pfdphase                Preferred-phase
PLV                     Phase-Locking Value
PSD                     Power Spectrum Density
powerPSD                Power exacted from PSD
SpectralEntropy         Spectral entropy (entropy extracted from PSD)
================        ==================================================================================

brainpipe also provide a bundle of tools for features

================        ==================================================================================
Function                Description
================        ==================================================================================
bandRef                 Get usual oscillation bands informations
findBandName            Get physiological name of a frequency band
findBandFcy             Get frequency band from a physiological name
cfcVec                  Generate cross-frequency vectors
cfcRndSignals           Generate signals artificialy coupled (great to test pac methods)
================        ==================================================================================

Classification
--------------
Bundle of functions to classify extracted features.

.. code-block:: python

    from brainpipe.classification import *

.. cssclass:: table-striped

================        ==================================================================================
Function                Description
================        ==================================================================================
defClf                  Define a classifier
defCv                   Define a cross-validation
classify                Classify features (either each one of them or grouping)
generalization          Generalization of decoding performance (generally, across time)
mf                      Multi-features procedure. Select the best combination of features
MFpipe                  Multi-features pipeline
================        ==================================================================================


Statistics
--------------
Bundle of functions to apply statistics.

.. code-block:: python

    from brainpipe.statistics import *

.. cssclass:: table-striped

==================      ==================================================================================
Function                Description
==================      ==================================================================================
bino_da2p               Get associated p-value of a decoding accuracy using a binomial law
bino_p2da               Get associated decoding accuracy of a p-value using a binomial law
bino_signifeat          Get significant features using a binomial law
perm_2pvalue            Get p-value from a permutation dataset
perm_metric             Get a metric (usefull for mastat)
perm_pvalue2level       Get p-associated level in a permutation distribution
perm_rndDatasets        Generate random dataset of permutations
perm_swap               Randomly swap ndarray (matricial implementation)
perm_rep                Repeat a ndarray of permutations (matricial implementation)
bonferroni              Multiple comparison: Bonferroni
fdr                     Multiple comparison: False Discovery Rate
maxstat                 Multiple comparison: Maximum statistic
circ_corrcc             Correlation coefficient between one circular and one linear random variable
circ_r                  Computes mean resultant vector length for circular data
circ_rtest              Computes Rayleigh test for non-uniformity of circular data
==================      ==================================================================================


Visualization
--------------
Bundle of functions to visualize results and make some <3 pretty plots <3.

.. code-block:: python

    from brainpipe.visual import *

.. cssclass:: table-striped

================        ==================================================================================
Function                Description
================        ==================================================================================
BorderPlot              Plot data and deviation/sem in transparency
addPval                 Add p-values to an existing plot
continuouscol           Plot lines with continuous color
addLines                Quickly add vertical and horizontal lines
tilerplot               Generate automatic 1D or 2D subplots with a lot of control
addPval                 Add significants p-value to an existing plot
rmaxis                  Remove ticks and axis of a existing plot
despine                 Despine axis of a existing plot
================        ==================================================================================


Tools
-----
This part provide a set complement

.. code-block:: python

    from brainpipe.tools import *

.. cssclass:: table-striped

================        ==================================================================================
Function                Description
================        ==================================================================================
study                   Manage your current study without carrying of path
savefile                Quickly save files using most common extensions
loadfile                Quickly load files using most common extensions
pdTools                 Some complement functions for pandas Dataframe (search, keep, remove)
ndsplit                 Split ndarray (works on odd and even dimensions)
ndjoin                  Join ndarray (works on odd and even dimensions)
p2str                   Transform a p-value to string (usefull to save files with corresponding p-value)
uorderlst               Get unique ordered elements from a list
================        ==================================================================================


