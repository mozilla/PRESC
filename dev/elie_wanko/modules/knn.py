#Imports
import numpy as np
import pandas as pd
import random
import math
import timeit

from sklearn.neighbors import KNeighborsRegressor

def classfier(data, cutoff = 10, n_nearest = 5):
    training_attributes = data.columns[:-1]
    testing_attribute = data.columns[-1]

    # Randomly shuffle the index of nba.
    random_indices = np.random.permutation(data.index)

    # Set a cutoff for how many items we want in the test set (in this case 1/10 of the items)
    test_cutoff = math.floor(len(data)/cutoff)

    # Generate the test and train set.
    test = data.loc[random_indices[1:test_cutoff]]
    train = data.loc[random_indices[test_cutoff:]]
    
    # time on
    start = timeit.default_timer()
    # Look at the n_nearest neighbors.
    knn = KNeighborsRegressor(n_neighbors=n_nearest)
    # Fit the model on the training data.
    knn.fit(train[training_attributes], train[testing_attribute])
    # time off    
    stop = timeit.default_timer()
    time = stop - start
    
    # Make point predictions on the test set using the fit model.
    predictions = knn.predict(test[training_attributes])

    # Get the actual values for the test set.
    actual = test[testing_attribute]

    # Compute the mean squared error of our predictions.
    mse = (((predictions - actual) ** 2).sum()) / len(predictions)
    
    
    return predictions, actual, mse, time
    