
# https://towardsdatascience.com/how-to-train-test-split-kfold-vs-stratifiedkfold-281767b93869

import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=3)

print(skf.get_n_splits(X, y))

StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    _train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]