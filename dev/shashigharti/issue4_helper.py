from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def cross_validation(estimator, X, y):
    """ This function takes estimator and data as input and does cross-validation tests.
    Args:
        estimator: Model to be trained
        X: Training Data(features columns)
        y: Labels for each row
    Returns:
        dataframe: list of test-train split results
    """  
    result_lst = list()       
    

    # Convert the list to dataframe
    result_df = pd.DataFrame(result_lst, columns = ['Ratio(Train Data)', 'Ratio(Test Data)', 'Accuracy', 'F1-Score'])
        
    print(result_df)
    return result_df