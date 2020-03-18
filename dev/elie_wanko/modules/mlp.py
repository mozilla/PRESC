#Imports
from numpy import random, array
from math import floor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def classifier(alpha=1e-5, cut_off=10, scaler=False, solver='adam', activation ='relu', *, data, input_size, output_size):
    """ Returns the predicted values computed from a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.
    
    Keyword arguments:
    data(dataframe) -- data set 
    input_size(int) -- number of observation, which will be refactor to 2/3 the total number of observations.
    output_size -- 
    cut_off(int) -- slices the data set into training and testing subsets (default = 10% for testing subset)
    alpha(float) -- regularization (L2 regularization) term which helps in avoiding overfitting by penalizing weights with large magnitudes. (default = 1e-5) 
    scaler(boolean) -- feature scaling 
    """
    # Split training and testing attributes
    training_attributes = data.columns[:-1]
    testing_attribute = data.columns[-1]

    # Randomly shuffle the index of nba.
    random_indices = random.permutation(data.index)
   
    # Set a cutoff for how many items we want in the test set (in this case 1/10 of the items)
    cut_off = floor(len(data)/cut_off)
    input_size = floor(len(data)*2/3)
    
    # Generate the test and train set.
    test_data = data.loc[random_indices[1:cut_off]]
    train_data = data.loc[random_indices[cut_off:]]
    
    # Scale data
    if scaler==True:
        scaler = StandardScaler()
        # Don't cheat - fit only on training data
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        # apply same transformation to test data
        test_data = scaler.transform(test_data)
        
    mlp_c = MLPClassifier(solver=solver, alpha=alpha, shuffle=False, hidden_layer_sizes=(input_size, output_size), early_stopping=True)
    
    # Fit the model on the training data.
    mlp_c.fit(train_data[training_attributes], train_data[testing_attribute])
    
    # Make point predictions and Get the actual values from the test set.
    pred = mlp_c.predict(test_data[training_attributes])
    true = array(test_data[testing_attribute])
    
    #Returns predicted(pred) and actual(true) values
    return pred, true
    
    


