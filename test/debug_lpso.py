import numpy as np
from sklearn.datasets import make_classification
import brainpipe
from brainpipe.classification import LeavePSubjectOut

n_features = 5

X1, Y1 = make_classification(n_features=n_features, n_redundant=0,
                             n_informative=1, n_clusters_per_class=1)

print(X1.shape, Y1.shape, Y1)

# LeavePSubjectOut?