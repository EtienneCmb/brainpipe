import numpy as np
import matplotlib.pyplot as plt

from brainpipe.classification import *
from brainpipe.visual import *

# Number of features :
n_features = 5   # Number of features

# Function for dataset creation :
def dataset_pear_subject(ntrials, suj=0):
    """Create a dataset for each subject. This little function
    will return x and y, the dataset and the label vector of each subject
    """
    spread = np.linspace(0, 0.5, n_features)
    class1 = np.random.uniform(size=(ntrials, n_features)) + spread
    class2 = np.random.uniform(size=(ntrials, n_features)) - spread
    x = np.concatenate((class1, class2), axis=0)
    # y = np.ravel([[k]*ntrials for k in np.arange(2)])
    y = suj * np.ones((x.shape[0],))#np.ravel([[k]*ntrials for k in np.arange(2)])
    print(x.shape, y.shape)
    return x, y

# Create a random dataset and a label vector for each subject
x_s1, y_s1 = dataset_pear_subject(20, 0)   # 20 trials for subject 1
x_s2, y_s2 = dataset_pear_subject(25, 1)   # 25 trials for subject 2
x_s3, y_s3 = dataset_pear_subject(18, 0)   # 18 trials for subject 3
x_s4, y_s4 = dataset_pear_subject(10, 1)   # 10 trials for subject 4

# Concatenate all datasets and vectors in list :
x = [x_s1, x_s2, x_s3, x_s4]
y = [y_s1, y_s2, y_s3, y_s4]

# Classification object :
lpso = LeavePSubjectOut(y, 4, pout=1, clf='lda', kern='linear') # Leave ONE-subject out (pout)

# # Run classification :
da, pvalue, daperm = lpso.fit(x, n_perm=20, method='label_rnd', n_jobs=1)

# # Display informations about features :
# # print(lpso.info.featinfo)
print(daperm)