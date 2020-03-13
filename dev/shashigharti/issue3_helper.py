from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_train_split(estimator, X, y):
    """ This function takes train data, labels as input and trains the SVM model.
    The function uses GridSearchCV for hyperparameter tuning for SVM

    Args:
        X: Training Data(features columns).
        y: Labels for each row
        param_grid (json): params for GridSearchCV (c, gamma, kernel)
        ratio: (optional) Split ratio for train and test data
    Returns:
        tuple: y_test, y_pred values
    """  
    result_lst = list()       
    
    for i in range(10, 100, 5):
        
        # Calculate Test/Train Ratio
        test_ratio = round(i/100 , 2)
        train_ratio = round(1 - test_ratio, 2)
        
        # Split the data based on the calculated test ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
        
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        result_lst.append([train_ratio, test_ratio, metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='weighted')])
        
        # Convert the list to dataframe
        result_df = pd.DataFrame(result_lst, columns = ['Ratio(Train Data)', 'Ratio(Test Data)', 'Accuracy', 'F1-Score']) 
        
    print(result_df)
    return result_df