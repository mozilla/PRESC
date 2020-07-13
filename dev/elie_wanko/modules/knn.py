# Imports
from numpy import random, array
from math import floor

from sklearn.neighbors import KNeighborsRegressor


def __init__():
    pass


def classifier(
    n_neighbors=5,
    weights="uniform",
    algorithm="auto",
    leaf_size=30,
    p=2,
    metric="minkowski",
    metric_params=None,
    n_jobs=None,
    threshold=None,
    *,
    subsets
):
    """ Returns the predicted values computed from a simple majority vote of the nearest neighbors of each point.
    
    Keyword arguments:
    data(dataframe) -- data set 
    cutoff(int) -- slices the data set into training and testing subsets (default = 10% for testing subset)
    n_nearest(int) -- nearest neighbors searches (default = 5)
    threshold(float, [0.0-1.0] -- at what probb will an observation default
    """
    # Look at the n_nearest neighbors.
    knn = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        metric_params=metric_params,
        n_jobs=n_jobs,
    )

    # Fit the model on the training data.
    knn.fit(subsets[0], subsets[2])

    # Make point predictions and Get the actual values from the test set.
    pred = knn.predict(subsets[1])
    true = array(subsets[3])

    # Converts
    if threshold != None:
        for i in range(len(pred)):
            if pred[i] >= threshold:
                pred[i] = 1
            else:
                pred[i] = 0
        pred = pred.astype(int)

    # Returns predicted(pred) and actual(true) values
    return pred, true
