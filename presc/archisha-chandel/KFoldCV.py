__author__ = 'Archisha Chandel'
__copyright__ = 'Copyright 2020, mozilla/PRESC'

# importing necessary libraries
import os
import pandas as pd
import numpy as np
import itertools
import pandas_profiling
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    KFold,
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve,
    confusion_matrix,
    mean_squared_error,
    f1_score,
)


# reading the data
def read_data(folder, file_name):
    """
    This function reads the csv file from the folder it is stored in 
    and returns a dataframe.
    
    Parameters: main directory, file name
    
    Returns: dataframe    
    """

    data = pd.read_csv("../../{}/{}.csv".format(folder, file_name))

    return data


# studying the data
def study_data(data):
    """
    This function helps know the various details of the data like, 
    number of null values, skewness, distribution, correlations, 
    missing values, interactions, data type, unique values etc.
    
    Parameters: dataframe
    
    Returns: report object
    """

    profile = pandas_profiling.ProfileReport(data)

    return profile


# pre-processing
def preprocessing(X_train, X_test):
    """
    This function sclaes all the features of train and test
    data making it ready for further processing.
    
    Parameters: dataframe
    
    Returns: dataframe
    """

    # instantiating StandardScaler
    sc = StandardScaler()

    # fitting on train
    X_train = sc.fit_transform(X_train)

    # transforming test
    X_test = sc.transform(X_test)

    return X_train, X_test


# splitting the data into train-test
def data_split(data, t_size, rndm_state):
    """
    This function performs train-test split on the dataframe
    depending on the test-size and random-state specified.
    
    Parameters: dataframe, test-size (int) and 
    random-state (int)
    
    Returns: six dataframes namely- X_train, y_train, X_test,
    y_test and features, target
    """

    # defining features (x) columns
    data_x = data.drop(columns=["label"], axis=1)

    # defining target (y) column
    data_y = data["label"]

    # performing train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=t_size, random_state=rndm_state
    )

    return X_train, X_test, y_train, y_test, data_x, data_y


# reducing the feature-space
def pca(X_train, X_test):
    """
    This function fits PCA instance on train data and 
    returns a dataframe with 2 components. It also
    fit_transforms the test data
    
    Parameters: dataframe (X_train), dataframe (X_test)
    
    Returns: dataframe (X_train), dataframe (X_test)
    """

    # instantiating PCA
    pca = PCA(n_components=2)

    # fitting on train data
    X_train = pca.fit_transform(X_train)

    # transforming X_test on the basis of fitted data
    X_test = pca.transform(X_test)

    return X_train, X_test


# training rfc using train-test-split
def rfc_model(X_train, y_train, X_test, y_test):
    """
    This function fits RFC model on the input dataframes,
    predicts and calculates the f1-score to check the 
    model performance
    
    Parameters: four dataframes (X_train, y_train,
                X_test, y_test)
    
    Returns: float (f1-score for training),
             classififctaion report,
             confusion matrix
    """

    # creating RFC instance
    rf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=50,)

    # fitting the model
    rf.fit(X_train, y_train)

    # predicting data
    y_pred = rf.predict(X_test)

    # calculating f1-score
    f1score = f1_score(y_test, y_pred, average="micro")
    print("==========================================")
    print("=================F1-Score=================")
    print("F1-Score: {}".format(f1score))

    # calculating classification report
    eval_report = classification_report(y_test, y_pred)
    print("==========================================")
    print("==========Classification Report===========")
    print(eval_report)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print("==========================================")
    print("=============Confusion Matrix=============")
    print(cnf_matrix)

    return f1score, eval_report, cnf_matrix


# training rfc using kfolds to split
def kfold(data_x, data_y, kfolds):
    """
    This function performs training on data using KFold
    and returns the mean of f1-score for the K-Folds
    
    Parameters: dataframe, int (kfolds)
    
    Returns: float (mean of K f1-scores)
    """

    # creating RFC instance
    rf = RandomForestClassifier(max_depth=4, random_state=46, n_estimators=20)

    # f1-score
    cv_f1_scores_rf = cross_val_score(rf, data_x, data_y, cv=kfolds, scoring="f1_micro")
    print("==========================================")
    print("=================F1-Score=================")
    print(cv_f1_scores_rf)
    print("Mean {}-Fold F1-Score: {}".format(kfolds, np.mean(cv_f1_scores_rf)))

    return np.mean(cv_f1_scores_rf)


def manual_kfold(data_x, data_y, kfolds):
    """
    This function manually performs kfold on the data
    
    Parameters: dataframe (features), dataframe (target),
                integer (kfolds)
                
    Returns: float (avg score of all K-folds)
    """

    # Defin the KFold class and train the model, am using K as 5
    kf = KFold(n_splits=kfolds, shuffle=True)
    rf = RandomForestClassifier(max_depth=4, random_state=46, n_estimators=20)

    # Perform the training repeatedly on each k-1 folds (x_train, y_train) and use the Kth fold (x_test, y_test) for testing. Append the
    # score from every iteration to the scores list.

    scores = []
    for i in range(kfolds):
        result = next(kf.split(data_x), None)
        x_train = data_x.iloc[result[0]]
        x_test = data_x.iloc[result[1]]
        y_train = data_y.iloc[result[0]]
        y_test = data_y.iloc[result[1]]
        model = rf.fit(x_train, y_train)
        predictions = rf.predict(x_test)
        scores.append(model.score(x_test, y_test))
    print("==========================================")
    print("===================Score==================")
    print("Scores from each Iteration: ", scores)
    print("Average K-Fold Score :", np.mean(scores))

    return np.mean(scores)


def score_folds(data_x, data_y):
    """
    This fucntion outputs a table and plot explaining
    the variation in evaluation metric over a range of 
    k-folds.
    
    Parameters: two dataframes (features and target)
    
    Return: int array (folds), float array (avg_score)
    """

    # initialising arrays
    kfolds = []
    avg_score = []

    # initialising model
    estimator = RandomForestClassifier(max_depth=4, random_state=46, n_estimators=20)

    # calucating avg_score per fold value
    for fold in range(2, 51):

        # calculating score
        scores = cross_val_score(estimator, data_x, data_y, cv=fold)
        mean_score = np.mean(scores)
        # update lists
        avg_score.append(mean_score)
        kfolds.append(fold)

    # Print table
    table = pd.DataFrame({"Number of folds": kfolds, "Average score": avg_score})
    print(table)

    # Plot the variation of average score as caused by changing the number of folds
    plt.figure()
    plt.plot(kfolds, avg_score, label="Average accuracy")
    plt.legend()
    plt.xlabel("Number of folds")
    plt.ylabel("Average accuracy")
    plt.show()

    return kfolds, avg_score
