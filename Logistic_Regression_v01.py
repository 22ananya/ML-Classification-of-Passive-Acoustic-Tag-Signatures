# Logistic Regression Approach for classifying Acoustic Signatures

# Load Dependencies
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat



## Data Load Function
file = 'Data\Dataset_v01.mat' # File name of Data File to be loaded
def load_data(filename):
    data = loadmat(filename)
    return data

data = load_data(file)
X, Y = np.transpose(np.array(data['X'])), np.transpose(np.array(data['Y']))
n,d = X.shape # d is data dimension, n is total number of samples
#X, Y = np.transpose(X,(n,d)), np.reshape(Y,(n,1))

# Plot random data sample to ensure it is correctly loaded and dimensioned
fig1, ax = plt.subplots()
ax.plot(X[np.random.randint(n),:])
ax.set_xlabel('Samples')
ax.set_ylabel('Amplitude')


## Build Logistic Regressor
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)



