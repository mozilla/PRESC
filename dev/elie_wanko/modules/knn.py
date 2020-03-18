#Imports
from numpy import random, array
from math import floor

from sklearn.neighbors import KNeighborsRegressor

def classfier(data, cut_off = 10, n_nearest = 5):
    """ Returns the predicted valuescomputed from a simple majority vote of the nearest neighbors of each point.
    
    Keyword arguments:
    data(dataframe) -- data set 
    cutoff(int) -- slices the data set into training and testing subsets (default = 10% for testing subset)
    n_nearest(int) -- nearest neighbors searches (default = 5)
    """
    
    # Split training and testing attributes
    training_attributes = data.columns[:-1]
    testing_attribute = data.columns[-1]

    # Randomly shuffle the index of nba.
    random_indices = random.permutation(data.index)

    # Set a cutoff for how many items we want in the test set (in this case 1/10 of the items)
    cut_off = floor(len(data)/cut_off)

    # Generate the test and train set.
    test_data = data.loc[random_indices[1:cut_off]]
    train_data = data.loc[random_indices[cut_off:]]

    # Look at the n_nearest neighbors.
    knn = KNeighborsRegressor(n_neighbors=n_nearest)
    # Fit the model on the training data.
    knn.fit(train_data[training_attributes], train_data[testing_attribute])
    
    # Make point predictions and Get the actual values from the test set.
    pred = knn.predict(test_data[training_attributes])
    true = array(test_data[testing_attribute])
    
    #Returns predicted(pred) and actual(true) values
    return pred, true
    