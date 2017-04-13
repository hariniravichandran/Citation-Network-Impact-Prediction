import numpy as np
from sklearn.metrics import *
from sklearn.neighbors import *

# Input - list of all papers with its features [[],[],..]
citation_data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y - class label - Prediction values
citations = np.array([0,1,2,3,4,5])
# Creates an instance of a model
model = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
# Fit the model for the given x, y
model.fit(citation_data, citations)
# Accuracy
test_data = [[0,0.9], [0, -0.1]]
test_citation = [1, 0]
pred = model.predict(test_data)
print accuracy_score(test_citation, pred)