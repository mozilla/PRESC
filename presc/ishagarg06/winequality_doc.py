# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# import the data
def read_csv(File_Path):
    """
    This function import the csv file from the path it is stored in
    and returns a dataframe.
    Parameters: Path of the dataset
    Returns: dataframe
    """
    data = pd.read_csv(File_Path)

    return data

# read the data

def study_data(data):
    """
    This function helps know the various details of the data like,
    min, max, count, std_deviation and mean of the values.
    Also number of null values, skewness, distribution, correlations,
    missing values, interactions, data type, unique values etc.
    Parameters: dataframe
    Returns: report object
    """

    return data

# pre-processing
def preprocessing(data):
    """
    This function drops duplicates present in the dataframe if present and
    converts the target column from object to int datatype.
    Parameters: dataframe
    Returns: dataframe
    """

    # dropping duplicates
    data.drop_duplicates(inplace=True)

    # converting data['recommend'] to integer column
    data.recommend = data.recommend * 1
    """
    The recommend column is converted to numeric from categorical data and
    False becomes --> 0
    True becomes --> 1
    """
    return data

# defining features (x) columns
data_x = data.drop(columns=["recommend"], axis=1)

# defining target (y) column
data_y = data["recommend"]

# splitting the data into training and test set
def data_split(data, tr_size, rndm_state):
    """
    This function performs train-test split on the dataframe
    depending on either train-size or test-size and random-state specified.
    Parameters: dataframe, train-size (int) or test-size (int) and
                random-state (int)
    Returns: four dataframes namely- X_train, y_train, X_test, y_test
    """
    # performing train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, train_size=tr_size, random_state=rndm_state
    )

    return X_train, X_test, y_train, y_test

def feature_scaling(X_train, X_test):
    """
    This function transforms the data such that its distribution
    will have a mean value 0 and standard deviation of 1.
    The main idea is to normalize/standardize features/variables/columns
    of X before applying machine learning techniques.

    Methods: fit, fit_transform
    Returns: X_train, X_test
    """
    std_sc = StandardScaler()
    X_train = std_sc.fit_transform(X_train)
    X_test = std_sc.transform(X_test)

    return X_train, X_test

# building various classification models

# DECISION TREE
def decision_tree_model(X_train, y_train, X_test):
    """
    This function go from observations about an item
    (represented in the branches) to conclusions about the
    item's target value (represented in the leaves).
    Parameters: three dataframes (X_train, y_train, X_test)
    Returns: model
    """
    # create object of decision tree
    decision_cls = DecisionTreeClassifier(criterion="entropy", random_state=100)
    """
    Parameters: criterion, random_state
    """
    # fit model on train dataset
    decision_cls.fit(X_train, y_train)

    # Predicting the test set results
    y_pred = decision_cls.predict(X_test)

    # making confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Accuracy score
    acc = accuracy_score(y_test, y_pred)
    acc

    return acc

# RANDOM FOREST CLASSIFICATION
def random_forest_model(X_train, y_train, X_test):
    """
    #This function  creates decision trees on randomly selected data
    samples, gets prediction from each tree and selects the best
    solution by means of n-estimators.
    #Parameters: three dataframes (X_train, y_train, X_test)
    #Returns: model
    """
    # create object of decision tree
    random_cls = RandomForestClassifier(
        n_estimators=2, criterion="entropy", random_state=100
    )
    """
    Parameters: n_estimators, criterion, random_state
    """
    # fit model on train dataset
    random_cls.fit(X_train, y_train)

    # Predicting the test set results
    y_pred2 = decision_cls.predict(X_test)

    # making confusion matrix
    cm = confusion_matrix(y_test, y_pred2)

    # Accuracy score
    acc2 = accuracy_score(y_test, y_pred2)
    acc2

    return acc2

# SUPPORT VECTOR MACHINE
def decision_tree_model(X_train, y_train, X_test):
    """
    This function depends on some subset of the training data,
    called the support vectors. It's a two-group classification model.
    Parameters: three dataframes (X_train, y_train, X_test)
    Returns: model
    """
    # create object of decision tree
    svm_cls = SVC(kernel="linear", random_state=100)
    """
    Parameters: kernel, random_state
    """
    # fit model on train dataset
    svm_cls.fit(X_train, y_train)

    # Predicting the test set results
    y_pred3 = decision_cls.predict(X_test)

    # making confusion matrix
    cm = confusion_matrix(y_test, y_pred3)

    # Accuracy score
    acc3 = accuracy_score(y_test, y_pred3)
    acc3

    return acc3
