#Imports
import numpy as np
import math

from sklearn.neighbors import KNeighborsRegressor

def classfier(data, cutoff = 10, n_nearest = 5):
    """ Returns the predicted valuescomputed from a simple majority vote of the nearest neighbors of each point.
    
    Keyword arguments:
    data(dataframe) -- data set 
    cutoff(int) -- slices the data set into training and testing subsets (default = 10% for testing subset)
    n_nearest(int) -- nearest neighbors searches (default = 5)
    """
    training_attributes = data.columns[:-1]
    testing_attribute = data.columns[-1]

    # Randomly shuffle the index of nba.
    random_indices = np.random.permutation(data.index)

    # Set a cutoff for how many items we want in the test set (in this case 1/10 of the items)
    test_cutoff = math.floor(len(data)/cutoff)

    # Generate the test and train set.
    test = data.loc[random_indices[1:test_cutoff]]
    train = data.loc[random_indices[test_cutoff:]]

    # Look at the n_nearest neighbors.
    knn = KNeighborsRegressor(n_neighbors=n_nearest)
    # Fit the model on the training data.
    knn.fit(train[training_attributes], train[testing_attribute])
    
    # Make point predictions and Get the actual values from the test set.
    pred = knn.predict(test[training_attributes])
    true = np.array(test[testing_attribute])
    
    #Returns predicted(pred) and actual(true) values
    return pred, true
    