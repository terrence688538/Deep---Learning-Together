import numpy as np
import sklearn.model_selection # import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
y = np.array([1, 2, 3, 4]) # Create another array
kf = sklearn.model_selection.KFold(n_splits=2) # Define the split - into 2 folds
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf)
for train_index, test_index in kf.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

kf = sklearn.model_selection.RepeatedKFold(n_splits=2, n_repeats=2, random_state=0)
for train_index, test_index in kf.split(X):
    print('train_index', train_index, 'test_index', test_index)

X = [1, 2, 3, 4]
loo = sklearn.model_selection.LeaveOneOut()
for train_index, test_index in loo.split(X):
    print('train_index', train_index, 'test_index', test_index)

X = [1, 2, 3, 4]
lpo=sklearn.model_selection.LeavePOut(p=2)
for train_index, test_index in lpo.split(X):
    print('train_index', train_index, 'test_index', test_index)

