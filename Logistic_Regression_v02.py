# Logistic Regression Approach for classifying Acoustic Signatures

                        ## Load Dependencies
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat

# test link
                        ## Data Load Function
file = 'Data\Dataset_v02.mat' # File name of Data File to be loaded
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

# Standardize data - Zero mean and Unit Variance - plot an example again to verify
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Plot random data sample to ensure it is correctly rescaled 
fig2, ax = plt.subplots()
ax.plot(X[np.random.randint(n),:])
ax.set_xlabel('Samples')
ax.set_ylabel('Amplitude')
ax.set_title('Scaled Example')

# load classifier and fit data
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression() # shorter function name

        ## Implementing Stratified K fold Validation
from sklearn.model_selection import StratifiedKFold
# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(n))
lst_accu_stratified = []

for train_index, test_index in skf.split(x_train, y_train):
    x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    lgr.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(lgr.score(x_test_fold, y_test_fold))


# Print list of training accuracy achieved
print(lst_accu_stratified)


                        ## Only Test on Hold Out Set when satisfied with CV results. Don't "train" using repeated test set evaluations!
# Predict labels
lgr.fit(x_train, y_train) # train again on complete training set before testing on hold out set
predictions = lgr.predict(x_test)

# metrics
from sklearn.metrics import accuracy_score
score = accuracy_score(predictions, y_test)
print(score)

# PLotting Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()