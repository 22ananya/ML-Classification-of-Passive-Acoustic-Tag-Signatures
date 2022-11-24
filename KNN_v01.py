# KNN (K Nearest Neighbor) Classifier for Acoustic Signatures

import numpy as np
from sklearn import neighbors

            ## Load Training Data ##

x_train = np.load('Data/numpy/matlab.mat_x_train.npy')
y_train = np.load('Data/numpy/matlab.mat_y_train.npy')

            ## Create KNN Classifier ##

# Assign k = number of neighbors to consider
k = 5
classifier = neighbors.KNeighborsClassifer(k, 'uniform')

