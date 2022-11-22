import numpy as np
import scipy
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import sys

print(f"Pre-processing {sys.argv[1]}...")
data = loadmat(sys.argv[1])
X, Y = np.transpose(np.array(data['X'])), np.ravel(np.array(data['Y']))
n,d = X.shape # d is data dimension, n is total number of samples

# Split Dataset
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Standardize data - Zero mean and Unit Variance - plot an example again to verify
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


np.save("Data/numpy/" + os.path.basename(sys.argv[1]) + "_x_train.npy", x_train)
np.save("Data/numpy/" + os.path.basename(sys.argv[1]) + "_x_test.npy", x_test)
np.save("Data/numpy/" + os.path.basename(sys.argv[1]) + "_y_train.npy", y_train)
np.save("Data/numpy/" + os.path.basename(sys.argv[1]) + "_y_test.npy", y_test)