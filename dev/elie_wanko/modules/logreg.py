# Imports
from numpy import random, array
from math import floor

from sklearn.linear_model import LogisticRegression


def __init__():
    pass


def classifier(
    penalty="l2",
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    random_state=None,
    solver="liblinear",
    max_iter=100,
    multi_class="auto",
    verbose=0,
    warm_start=False,
    n_jobs=None,
    l1_ratio=None,
    *,
    subsets,
    data
):

    """ Returns the predicted values computed from a simple majority vote of the nearest neighbors of each point.
    
    Keyword arguments:
    data(dataframe) -- data set 
    cutoff(int) -- slices the data set into training and testing subsets (default = 10% for testing subset)
    n_nearest(int) -- nearest neighbors searches (default = 5)
    threshold(float, [0.0-1.0] -- at what probb will an observation default
    """

    # Look at the n_nearest neighbors.
    lg = LogisticRegression(
        penalty=penalty,
        dual=dual,
        tol=tol,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=random_state,
        solver=solver,
        max_iter=max_iter,
        multi_class=multi_class,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
        l1_ratio=l1_ratio,
    )

    # Fit the model on the training data.
    lg.fit(subsets[0], subsets[2])

    # Make point predictions and Get the actual values from the test set.
    pred = lg.predict(subsets[1])
    true = array(subsets[3])

    return pred, true
