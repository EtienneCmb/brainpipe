Import features:

.. code-block:: python

    from brainpipe.features import *

.. todo:: Coherence // permutation entropy // wavelet filtering (numpy.wlt)

Presentation
============
In order to classify diffrent conditions, you can extract from your neural signals, a large variety of features. The aim of a feature is to verify if it contains the main information you want to classify. For example, let's say you search if an electrode is accurate to differenciate resting state from motor behavior. You can for example extract beta or gamma power from this electrode and classify your resting state versus motor using power features. Here's the list of all the implemented features:

* :ref:`sigfilt`
* :ref:`amplitude` 
* :ref:`power`
* :ref:`tf`
* :ref:`phase`
* :ref:`plf`
* :ref:`pac`
* :ref:`pp`
* :ref:`erpac`
* :ref:`plp`
* :ref:`plv`
* :ref:`psd`
* :ref:`powpsd`
* :ref:`sentr`

Filtering based
===============
Those following features use filtering method to extract informations in specifics frequency bands

.. toctree::
   :maxdepth: 3

   filtfeat


Coupling features
=================
Those following features use coupling (either distant or locals coupling)

.. toctree::
   :maxdepth: 3

   couplingfeat


PSD based features
==================
Those following features are extracted using a Power Spectrum Density (PSD)

.. toctree::
   :maxdepth: 3

   psdfeat


Tools
=====

.. toctree::
   :maxdepth: 3

   featools






