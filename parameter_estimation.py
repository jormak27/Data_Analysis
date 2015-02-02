#Jordan Makansi
#CEE 263N - Final Project
#12/9/14

#imports
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import spatial
from sklearn import cross_validation
#from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


#Global Variables and constants 
H_MIN = 0.01  #minimum bandwidth tested 
H_MAX = 3 #maximum bandwidth tested 
STEP_SIZE_H = 0.01  #step size for H iterations 
SIGMA_2_MIN = 0.01  # minimum covariance variance tested 
SIGMA_2_MAX = 0.1 # maximumcovariance tested 
STEP_SIZE_SIGMA = 0.01 # step size for covariance iterations

OPTIMAL_H = H_MIN 
OPTIMAL_SIGMA_2 = SIGMA_2_MIN
GRANULARITY = 50


### Loading the Digits dataset
digits = datasets.load_digits()
 
sys.exit(1)

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
