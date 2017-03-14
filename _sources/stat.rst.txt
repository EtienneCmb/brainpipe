.. code-block:: python

    from brainpipe.statistics import *


* :ref:`bino`
* :ref:`perm` 
* :ref:`mltpcomp`
* :ref:`circstat`


.. _bino:

Binomial
========
.. autofunction:: statistics.bino_da2p

.. autofunction:: statistics.bino_p2da

.. autofunction:: statistics.bino_signifeat

.. _perm:

Permutations
============
Evaluation
----------
.. autofunction:: statistics.perm_2pvalue

.. autofunction:: statistics.perm_metric

.. autofunction:: statistics.perm_pvalue2level

Generate
--------
.. autofunction:: statistics.perm_rndDatasets

.. autofunction:: statistics.perm_swap

.. autofunction:: statistics.perm_rep

.. _mltpcomp:

Multiple-comparisons
====================
Bonferroni
----------
.. autofunction:: statistics.bonferroni

False Discovery Rate (FDR)
--------------------------
.. autofunction:: statistics.fdr

Maximum statistic
-----------------
.. autofunction:: statistics.maxstat

.. _circstat:

Circular statistics toolbox
===========================
Python adaptation of the Matlab toolbox (Berens et al, 2009) 

.. autofunction:: statistics.circ_corrcc

.. autofunction:: statistics.circ_r

.. autofunction:: statistics.circ_rtest



