# Imports
from numpy import random, array
from math import floor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def __init__():
    pass
    
def classifier(alpha=1e-5, cut_off=10, scaler=False, solver="adam", activation="relu", *, 
               subsets, input_size, output_size):
    """ Returns the predicted values computed from a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
    
    Keyword arguments:
    data(dataframe) -- data set 
    input_size(int) -- number of observation, which will be refactor to 2/3 the total number of observations.
    output_size -- 
    cut_off(int) -- slices the data set into training and testing subsets (default = 10% for testing subset)
    alpha(float) -- regularization (L2 regularization) term which helps in avoiding overfitting by penalizing weights with large magnitudes. (default = 1e-5) 
    scaler(boolean) -- feature scaling 
    """
    # Scale data
    if scaler == True:
        scaler = StandardScaler()
        # Don't cheat - fit only on training data
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        # apply same transformation to test data
        test_data = scaler.transform(test_data)

    mlp_c = MLPClassifier(
        solver=solver,
        alpha=alpha,
        shuffle=False,
        hidden_layer_sizes=(input_size, output_size),
        early_stopping=True,
    )

    # Fit the model on the training data.
    mlp_c.fit(subsets[0], subsets[2])

    # Make point predictions and Get the actual values from the test set.
    pred = mlp_c.predict(subsets[1])
    true = array(subsets[3])

    # Returns predicted(pred) and actual(true) values
    return pred, true
