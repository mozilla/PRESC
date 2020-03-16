from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
import warnings

def cross_validation(estimator, X, y):
    """ This function takes estimator and data as input and does cross-validation tests.
    Args:
        estimator: Model to be trained
        X: Training Data(features columns)
        y: Labels for each row
    Returns:
        dataframe: list of test-train split results
    """  
    return ''

def cross_validation(X, y, param_grid, scaler = None, ratio = 0.4):
    """ This function takes train data, labels as input and trains the SVM model.
    The function uses GridSearchCV for hyperparameter tuning for SVM
    
    Args:
        X: Training Data(features columns).
        y: Labels for each row
        param_grid (json): params for GridSearchCV (c, gamma, kernel)
        scaler: Transformation class object
        ratio: (optional) Split ratio for train and test data
    Returns:
        tuple: y_test, y_pred values
    """
    warnings.warn("deprecated", DeprecationWarning)
    
    if scaler is not None:
        X = scaler.fit_transform(X)
        print(type(scaler))

    svm_grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = False) 
    #svm_grid.fit(X_train, y_train) 
    #y_pred = svm_grid.predict(X_test) 
    
    
    #10 different test 
    score = cross_val_score(svm_grid, X, y, cv=10)
    print(score.mean())

    
    return True