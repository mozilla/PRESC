# importing necessary libraries
import os
import pandas as pd
import numpy as np
import itertools
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


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


# splitting the data
def data_split(data, t_size, label):
    """
    This function performs train-test split on the dataframe
    depending on the test-size and random-state specified.
    
    Parameters: dataframe, test-size (int), 
    label (string) which represents the target column in 
    the dataset being used.
    
    Returns: six dataframes namely- X_train, y_train, X_test,
    y_test and train, test
    """

    # defining features (x) columns
    data_x = data.drop(columns=[label], axis=1)

    # defining target (y) column
    data_y = data[label]

    # performing train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=t_size, random_state=46, shuffle=True,
    )

    # train set
    train = pd.concat([X_train, y_train], axis=1)

    # test set
    test = pd.concat([X_test, y_test], axis=1)

    return X_train, X_test, y_train, y_test, train, test


# pre-processing on data
def preprocessing(train, test, X_train, y_train):
    """
    This function adds a new column and helps visualise the
    importance of individual features in building the model. The new
    column added indicates if the data point belongs to train set or
    test set.
    
    Parameters: four dataframes (train, test, X_train and y_train)
    
    Returns: two dataframes (train and test set)
    """

    # data point does not belong to train
    test["is_train"] = 0

    # data point belongs to train
    train["is_train"] = 1

    # using RFC to visualise top 5 fetaures in model training
    clf = RandomForestClassifier(
        criterion="gini", max_depth=8, max_features=6, n_estimators=300, random_state=0,
    )
    clf.fit(X_train, y_train)

    # studying top 5 important features
    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(5).plot(kind="barh")
    plt.title("Top 5 important features")
    plt.show()

    return train, test


# combining train and test after preprocessing
def combine_train_test(train, test, label):
    """
    This function combines the train and test data after performing
    pre-processing and dropping traget column as the values for it
    are not present in test set. It also divides the combined data
    into features and target
    
    Parameters: two dataframes (train adn test), string representing
    the target column name
    
    Returns: two dataframes (features and target),
    series (train values and test values)
    """

    # combining train, test
    df_combine = pd.concat([train, test], axis=0, ignore_index=True)

    # dropping ‘target’ column as it is not present in the test
    df_combine = df_combine.drop(label, axis=1)

    y = df_combine["is_train"].values  # labels
    x = df_combine.drop("is_train", axis=1).values  # independent variables

    tst, trn = test.values, train.values

    return x, y, tst, trn


# performing stratified KFold train test split on data obtained from combine_train_tets
def stratkf_split(x, y):
    """
    This function performs stratifiedKFold splitting on the data,
    fits RFC on the train data, predicts on test data and prints the
    roc_auc_score
    
    Parameters: two dataframes (x and y)
    
    Returns: array of the predictions made
    """

    # Building and testing a classifier
    rfc = RandomForestClassifier(n_jobs=3, max_depth=5, min_samples_leaf=5)

    # creating an empty prediction array
    predictions = np.zeros(y.shape)

    # initializing model
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=46)
    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
        X_train, X_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rfc.fit(X_train, y_train)
        probs = rfc.predict_proba(X_test)[:, 1]  # calculating the probability
        predictions[test_idx] = probs

    print("ROC-AUC:", roc_auc_score(y_true=y, y_score=predictions))

    return predictions


# assigning weights to individual data points
def calc_weights(predictions, trn):
    """
    This function assigns weights to individual data points and helps
    visualise the distribution of number of samples having the
    computed weights.
    
    Parameters: array of predictions computed above and train dataframe
    
    Returns: array of weights
    """

    plt.figure(figsize=(20, 10))

    # filtering the actual training rows
    predictions_train = predictions[: len(trn)]

    # defining weights
    weights = (1.0 / predictions_train) - 1.0
    weights /= np.mean(weights)  # Normalizing the weights

    # visualising #samples having the computed weights
    plt.xlabel("Computed sample weight")
    plt.ylabel("# Samples")
    sns.distplot(weights, kde=False)

    return weights


# passing the weights and running the model
def rfc_model(weights, X_train, y_train, X_test, y_test):
    """
    This function runs RFC with weights calculated from above method
    for training and tests it as well displaying the classification
    report and confusion matrix.
    
    Parameters: array of weights computed above,
    four dataframes describing the train-test split
    
    Returns: None
    """

    # instantiating model
    m = RandomForestClassifier(
        n_jobs=3, min_samples_split=13, n_estimators=300, max_depth=10, max_features=13
    )

    # fitting on train data
    m.fit(X_train, y_train, sample_weight=weights[: len(X_train)])

    # predicting on test data
    y_pred = m.predict(X_test)

    # calculating classification report
    eval_report = classification_report(y_test, y_pred)
    print(eval_report)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print(cnf_matrix)
