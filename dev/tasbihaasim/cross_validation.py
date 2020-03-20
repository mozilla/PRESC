from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
import random
import numpy as np

def cross_validation(X, Y, y_t):
    '''returns the accuracy score of the trained classification model

    args:
        - X: datapoints
        - Y: classes of the datapoints
    '''

    # set the number of repetitions
    times = 50

    # loop for cross validation
    for i in range(times):
        y_pred = np.zeros(times)

        kf = KFold(n_splits=5 , random_state=200)
        for train, test in kf.split(X):

            # make testing sets and training sets out of X and Y
            train_X = X.iloc[train]
            train_Y = Y.iloc[train]
            test_X = X.iloc[test]        
            test_Y = Y.iloc[test]

            # training model using the data set
            clf = KNeighborsClassifier(n_neighbors=5)
            clf.fit(train_X, train_Y)

            # making predictions
            y_pred = clf.predict(test_X)
        accuracy_score_x = accuracy_score(y_t, y_pred)


    return(accuracy_score_x) 
