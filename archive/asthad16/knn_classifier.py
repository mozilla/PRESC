#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def grid_knn(X, y):
    """for hyper parameter tuning to choose appropriate parameter values to improve the overall accuracy of our classifier"""
    k_range = list(range(1, 31))
    weight_options = ["uniform", "distance"]
    # create a dictionary of all values we want to test
    param_grid = dict(n_neighbors=k_range, weights=weight_options)
    # knn model
    knn = KNeighborsClassifier()
    # use gridsearch to test all values
    grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")
    # fit model to data
    grid.fit(X, y)
    return grid.best_params_


def knn_class(X_test, X_train, y_train, k, weight):
    """this function uses k-nearest neighbour algorithm.it creates an imaginary boundary to classify the data. 
    When new data points come in, the algorithm will try to predict that to the nearest of the boundary line.
    
    parameter - n_neighbors(no. of data points considered as neighbour) and weights(method for choosing most close neighbour) 
    tuned using gridsearch CV"""
    classifier = KNeighborsClassifier(n_neighbors=k, weights=weight)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob)
