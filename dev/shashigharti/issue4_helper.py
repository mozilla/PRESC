from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd
import warnings

def fix_outlier_with_boundary_value(df, df_columns_with_outliers): 
    """ This function takes data and outlier columns as input and replaces outliers > max value of each columns by 
    max value of that feature column
    
    Args:
        df: dataframe of all data
        df_columns_with_outliers: list of column names with outliers
    Returns:
        data_df: processed dataframe        
    """
    
    data_df = df.copy()

    # Fill null
    data_df.fillna(data_df.mean(), inplace=True)

    # Replace outliers with max boundary value
    for i, column in enumerate(df_columns_with_outliers.columns):
        data_df.loc[data_df[column] > df_columns_with_outliers[column][0], column] = df_columns_with_outliers[column][0]
        
    data_df['Class'] = pd.Categorical(data_df['Class']).codes
    
    return data_df

def test_cross_validation(estimator, X, y):
    """ This function takes estimator and train data as params and runs training on estimator with different
    splits(K-Fold) value. And it returns the result in tabular format 
    
    Args:
        estimator: Model to train
        X: Training Data(features columns).
        y: Labels for each row
    Returns:
        experiment_results(dataframe): table of results with K-Fold value and Avg. Score in each row 
    """
    
    scaler = MinMaxScaler()
     
    experiment_results = list()
    
    # 10 Passess from 5 - 15
    for splits in range(5, 15, 1):
        score = stratified_cross_validation(estimator, X, y, scaler, splits)
        # print("{}-Fold CV score:{}".format(splits, score.mean()))
        experiment_results.append([splits, round(score.mean(), 2)])
        
    # Convert the list to dataframe
    experiment_results = pd.DataFrame(experiment_results, columns = ['K-Fold CV', 'Avg. Score']) 
    return experiment_results

def stratified_cross_validation(estimator, X, y, scaler, splits, ratio = 0.4):
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
    warnings.filterwarnings('ignore')   
    
    # Stratified Cross Validation, n_splits is the number of folds
    # random_state is set to None as it won't do anything because shuffle is not set to True.
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
    skf = StratifiedKFold(n_splits=splits, random_state=None) 
    
    result_lst = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Data transformation
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        
        estimator.fit(X_train, y_train) 
        y_pred = estimator.predict(X_test) 
        
        result_lst.append([accuracy_score(y_pred, y_test)])
     
    
    return np.array(result_lst)
