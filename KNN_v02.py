# KNN (K Nearest Neighbor) Classifier for Acoustic Signatures

import numpy as np
from sklearn import neighbors

            ## Load Training Data ##

data_path = 'Data/numpy/'

x_train = np.load(data_path + 'Dataset_v02_x_train.npy')
y_train = np.load(data_path + 'Dataset_v02_y_train.npy')
n,d = x_train.shape
print(f'x-train shape: {x_train.shape}')
print(f'y-train shape: {y_train.shape}')

            ## Create and Optimize KNN Classifier ##

# Assign k = number of neighbors to consider
print('k | Average Accuracy')
print('--------------------')
for k in range(1,2):
    knn = neighbors.KNeighborsClassifier(k, weights='uniform')

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(n))
    knn_accu_stratified = []

    for train_index, test_index in skf.split(x_train, y_train):
        x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        knn.fit(x_train_fold, y_train_fold)
        knn_accu_stratified.append(knn.score(x_test_fold, y_test_fold)*100)
    
    print(f'k = {k}: {np.mean(knn_accu_stratified):.2f}%')

            ## Choose Best K and Train on Full Training Set

k = 1 
knn = neighbors.KNeighborsClassifier(k, weights='uniform')
knn.fit(x_train, y_train)

            ## Load Testing Data ##

x_test = np.load(data_path + 'Dataset_v02_x_test.npy')
y_test = np.load(data_path + 'Dataset_v02_y_test.npy')
print(f'x-test shape: {x_test.shape}')
print(f'y-test shape: {y_test.shape}')

            ## Make Predictions and Check Accuracy ##

predictions = knn.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predictions, y_test)
print(f'{accuracy*100:.2f}%')
