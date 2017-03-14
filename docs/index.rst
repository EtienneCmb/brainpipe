Brainpipe
*********

Brainpipe is a python toolbox dedicated for neuronal signals analysis and machine-learning. The aim is to provide a variety of tools to extract informations from neural activities (features) and use machine-learning to validate hypothesis. The machine-learning used the excellent `scikit-learn <http://scikit-learn.org/stable/>`_  library. Brainpipe can also perform parallel computing and try to optimize RAM usage for our 'large' datasets. If you want to have a quick overview of what you can actually do, checkout the :ref:`refpart`.

It's evolving every day! So if you have problems, bugs or if you want to collaborate and add your own tools, contact me at e.combrisson@gmail.com

	

Requirement
***********
brainpipe is developed on Python 3, so the compatibility with python 2 is not guaranted! (not tested yet)

Please, check if you have this toolbox installed and already up-to-date:

- matplotlib (visualization)
- scikit-learn (machine learning)
- joblib (parallel computing)
- scipy
- numpy
- psutil

Installation
************

Ru the following commands :

.. code-block:: bash

	git clone git@github.com:EtienneCmb/brainpipe.git

Then, go to the brainpipe cloned folder and run :

.. code-block:: python

   python setup.py install

What's new
**********
v0.1.8
=======
- Fix LeavePSubjectOut permutations for unique labels.
- Fix classification plot colors.

v0.1.6
=======
- Fix scikit-learn v0.18 compatibility
- Add mutli-features pipelines
 
v0.1.5
=======
- classify: fit_stat() has been removed, there is only fit() now. confusion_matrix() has been rename to cm()
- LeavePSubjectOut: new leave p-subjects out cross validation
- classification: embedded visualization (daplot, cmplot), statistics (classify.stat.), dataframe for quick settings summarize (classify.info.) with excel exportation
- New folder: ipython notebook examples
- tools: unique ordered function


v0.1.4
=======
- PAC: new surrogates method (shuffle amplitude time-series) + phase synchro
- New features (phase-locked power (coupling), ERPAC (coupling), pfdphase (coupling), Phase-Locking Value (coupling), Phase-locking Factor (PLF in spectral) and PSD based features)
- New tools for physiological bands definition
- New plotting function (addPval, continuouscol)
- Start to make the python adaptation of circstat Matlab toolbox
- Add contour to plot2D() and some other parameters 
- New doc ! Checkout the :ref:`refpart`
- Bug fix: pac phase shuffling method, coupling time vector, statictical evaluation of permutations


v0.1.0
=======

- Statistics:
	- Permutations: array optimized permutation module
	- p-values on permutations can be compute on 1 tail (upper and lower part) or two tails
	- metric: 
	- Multiple comparison: maximum statistique, Bonferroni, FDR
- Features:
	- sigfilt//amplitude//power//TF: wilcoxon/Kruskal-Wallis/permutations stat test (comparison with baseline)
	- PAC: new Normalized Direct PAC method (Ozkurt, 2012)
- Visualization:
	- tilerplot() with plot1D() and plot2D() with automatic control of subplot
- Tools:
	- Array: ndsplit and ndjoin method which doesn't depend on odd/even size (but return list)
	- squarefreq() generate a square frequency vector
	

Organization
************
.. toctree::
   :maxdepth: 4
   :caption: PRE-PROCESSING

   preprocessing

.. toctree::
   :maxdepth: 4
   :caption: FEATURES

   feature

.. toctree::
   :maxdepth: 3
   :caption: CLASSIFICATION

   classification

.. toctree::
   :maxdepth: 3
   :caption: STATISTICS

   stat

.. toctree::
   :maxdepth: 3
   :caption: VISUALIZATION

   visualization

.. toctree::
   :maxdepth: 3
   :caption: OTHERS

   tools

.. toctree::
   :maxdepth: 4
   :caption: REFERENCES

   references



Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

