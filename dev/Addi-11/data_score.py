# In this file 2 methods are used, to  evaluate  the importance of a datapoint in the dataset

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from classifiers import Classifier

# =============== METHOD - 1 =============
# the point misclassified by all classifiers -- outliers
# the point misclassified by some and not by others -- has imp. info
# the point correctly classified by all -- easy point

def cal_misclassified(x_train, x_val, y_train, y_val):
    '''
    Calculates the misclassified values of each classifier

    Parameters:
        x_train:
        x_val:
        y_train:
        y_val:

    Returns:
        err_indexes:
    '''
    # list of all classifiers
    classifiers = ['KNeighbors','Random_Forest', 'svm_classifier', 'Gaussian', 'Decision_Tree', 'Logistic_Reg']
    model = Classifier()

    # create dictionay to store misclassified values for each classifier
    err_indexes = {}
    correct_indexes = {}
    wt = {}

    for clf in classifiers:
        train_clf = getattr(model, clf)(x_train, y_train)
        y_score = train_clf.predict(x_val)
        # get the indexes of misclassified values
        err_indexes[clf] = np.where(y_score != y_val)
        correct_indexes[clf] = np.where(y_score == y_val)
        # associate wt to each model, based on its accuracy
        acc = accuracy_score(y_val, y_score)
        wt[clf] = 1/(1-acc)

    # calculating outliers
    outliers = err_indexes['KNeighbors']

    for clf in classifiers:
        outliers = np.intersect1d(outliers, err_indexes[clf])

    # calculating points with trivial info 
    # print('Weights associated with each model :', wt)

    # correctly by k-nn but not by random-forest
    pt1 = np.intersect1d(correct_indexes['KNeighbors'], err_indexes['Random_Forest'])
    
    # correctly by random-forest not by decision-tree
    pt2 = np.intersect1d(correct_indexes['Random_Forest'], err_indexes['Decision_Tree'])

    # correctly by decision-tree not by svm
    pt3 = np.intersect1d(correct_indexes['Decision_Tree'], err_indexes['svm_classifier'])

    point_score = {'points with high info': pt1,
              'points with medium info' : pt2,
              'points with ': pt3}

    # calculating easy-points
    easy_points = correct_indexes['KNeighbors']
    for clf in classifiers:
        easy_points = np.intersect1d(easy_points, correct_indexes[clf])

    return outliers, point_score, easy_points

# =============== METHOD - 2 =============
# weighted models 

def weighted(clf, x_train, y_train):
    # wts = 
    model = Classifier()
    train_clf = getattr(model, clf)(x_train, y_train)
