## Classification using Neural Networks

                        ## Load Dependencies
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf


                        ## Data Load Function
file = 'Data\Dataset_v02.mat' # File name of Data File to be loaded
def load_data(filename):
    data = loadmat(filename)
    return data

data = load_data(file)
X, Y = np.transpose(np.array(data['X'])), np.ravel(np.array(data['Y']))
Y = Y - 1 # Get label range to [0, 8)
n,d = X.shape # d is data dimension, n is total number of samples

# Plot random data sample to ensure it is correctly loaded and dimensioned
fig1, ax = plt.subplots()
ax.plot(X[np.random.randint(n),:])
ax.set_xlabel('Samples')
ax.set_ylabel('Amplitude')

                            ## Pre-Processing

# Split Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=True)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Standardize data - Zero mean and Unit Variance - plot an example again to verify
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

                            ## Build NN Classifier model

model = tf.keras.models.Sequential([
  tf.keras.Input(shape = (d,)),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(8)
])

model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

                            ## Train NN 

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test,  y_test, verbose=2)

