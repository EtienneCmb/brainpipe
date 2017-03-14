.. code-block:: python

    from brainpipe.classification import *

Presentation
============
Ok, let's say you already have extracted features from your neural activity and now, you want to use machine-learning to verify if your features can discriminate some conditions. For example, you want to discriminate conscious versus unconscious people using alpha power on 64 EEG electrodes. Your data can be organized like this:

.. code-block:: python

	# Consider that conscious data have 150 trials and 130 for unconscious. So if you
	# print the shape of both, you'll have :
	print(conscious_data.shape, unconscious_data.shape)
	# ( (150, 67), (130, 67) )
	# Let's build your data matrix by concatenating along the trial dimension:
	x = np.concatenate( (conscious_data, unconscious_data), axis=0 )
	print('New shape of x: ', x.shape)
	# New shape of x: (280, 67)
	# Now, build your label vector to indicate to machine learning
	# which trial belong to which condition. We are going to use
	# 0 for conscious / 1 for unconscious. Finally, the label vector
	# will have the same length as the number of trials in x :
	y = [0]*conscious_data.shape[0] + [1]*unconscious_data.shape[0]

Now, we have the concatenated data and the label vector. To start using machine learning, we need two things:

- a classifier
- a cross-validation

In brainpipe, use defClf() to construct your classifier. Use defCv() to construct the cross-validation. Finally, the classify() function will linked this two objectsin order to classify your conditions. 

.. code-block:: python

	# Define a 50 times 5-folds cross-validation :
	cv = defCv(y, cvtype='kfold', rep=50, n_folds=5)
	# Define a Random Forest with 200 trees :
	clf = defClf(y, clf='rf', n_tree=200, random_state=100)
	# Past the two objects inside classify :
	clfObj = classify(y, clf=clf, cvtype=cv)
	# Evaluate the classifier on data:
	da = clfObj.fit(x)


Define a classifier
===================
.. automodule:: classification
   :members: defClf
   :noindex:

Define a cross-validation
=========================
.. automodule:: classification
   :members: defCv
   :noindex:

Classify
========
.. automodule:: classification
   :members: classify
   :noindex:

Leave p-subjects out
====================
.. automodule:: classification
   :members: LeavePSubjectOut
   :noindex:


Generalization
==============
.. automodule:: classification
   :members: generalization
   :noindex:

.. figure::  images/tg.png
   :align:   center

   Time-generalization using two features (alpha and gamma power)


Multi-features
==============
.. automodule:: classification
   :members: mf
   :noindex:

.. automodule:: classification
   :members: MFpipe
   :noindex:
