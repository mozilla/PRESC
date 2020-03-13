from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def cross_validation(estimator, X, y):
    """ This function takes estimator and data as input

    Args:
        estimator: 
        X: Training Data(features columns).
        y: Labels for each row
        
    Returns:
        list: list with test results
    """  
    result_lst = list()       
    
    
    return result_lst