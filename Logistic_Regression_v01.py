# Logistic Regression Approach for classifying Acoustic Signatures

                        ## Load Dependencies
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
X, Y = np.transpose(np.array(data['X'])), np.ravel(np.array(data['Y']))
n,d = X.shape # d is data dimension, n is total number of samples

# Plot random data sample to ensure it is correctly loaded and dimensioned
fig1, ax = plt.subplots()
ax.plot(X[np.random.randint(n),:])
ax.set_xlabel('Samples')
ax.set_ylabel('Amplitude')


                        ## Build Logistic Regressor
# Split Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# load classifier and fit data
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression() # shorter function name

# Fit data
lgr.fit(x_train, y_train)

# Predict labels
predictions = lgr.predict(x_test)

# metrics
from sklearn.metrics import accuracy_score
score = accuracy_score(predictions, y_test)
print(score)