from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


def test_train_split(estimator, X, y, scaler = None):
    """ This function takes estimator and data as input and does test-train split test in multiple passes.

    Args:
        estimator: Model to be trained
        X: Training Data(features columns)
        y: Labels for each row

    Returns:
        dataframe: list of test-train split results
    """  
    result_lst = list()       
    
    for i in range(10, 100, 5):
        
        # Calculate Test/Train Ratio
        test_ratio = round(i/100 , 2)
        train_ratio = round(1 - test_ratio, 2)
        
        # Split the data based on the calculated test ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
        
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        result_lst.append([train_ratio, test_ratio, metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred, average='weighted')])
        
        # Convert the list to dataframe
        result_df = pd.DataFrame(result_lst, columns = ['Ratio(Train Data)', 'Ratio(Test Data)', 'Accuracy', 'F1-Score']) 
        
    print(result_df)
    
    return result_df