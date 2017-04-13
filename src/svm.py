import numpy as np
from sklearn import svm
from sklearn.metrics import *

# Input - list of all papers with its features [[],[],..]
# x = [[1, 1], [1, -1],[-1, -1],[-1, 1]]
x = [[1,2], [5,8], [1.5,1.8], [8,8], [1,0.6], [9,11]]
x = np.array(x)
# Y - class label - Prediction values
y = [0, 1, 2, 3, 4, 5]
y = np.array(y)
# Creates an instance of a model
svm_model = svm.SVC(kernel='linear', C = 1.0)
# Fit the model for the given x, y
svm_model.fit(x, y)
# Accuracy
pred = svm_model.predict([[0.58,0.76], [1, 0.5], [10.58,10.76]])
print pred
test_citation = [4, 4, 6]
print accuracy_score(test_citation, pred)


