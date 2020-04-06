__author__ = 'Archisha Chandel'
__copyright__ = 'Copyright 2020, mozilla/PRESC'

# importing necessary libraries
import os
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    roc_curve,
    confusion_matrix,
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

    # setting index for the data
    data.set_index("id", inplace=True)

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


# performing preprocesing steps
def preprocessing(data):
    """
    This function performs preprocessing on the data studied so far
    using pandas profiling and works only on defaults.csv.

    Parameters: dataframe

    Returns: dataframe
    """

    # converting columns= [sex, education, pay_0...pay_6] to obejct

    # sex int --> object
    data["sex"].astype(object)

    # education int --> object
    data["education"].astype(object)

    # pay_* int --> object
    data["pay_0"].astype(object)
    data["pay_2"].astype(object)
    data["pay_3"].astype(object)
    data["pay_4"].astype(object)
    data["pay_5"].astype(object)
    data["pay_6"].astype(object)

    # generate binary values using get_dummies
    dum_df = pd.get_dummies(
        data,
        columns=[
            "sex",
            "education",
            "pay_0",
            "pay_2",
            "pay_3",
            "pay_4",
            "pay_5",
            "pay_6",
        ],
    )

    return dum_df


# splitting the data into train-test
def data_split(data, t_size, rndm_state):
    """
    This function performs train-test split on the dataframe
    depending on the test-size and random-state specified.

    Parameters: dataframe, test-size (int) and random-state (int)

    Returns: four dataframes namely- X_train, y_train, X_test, y_test
    """

    # defining features (x) columns
    data_x = data.drop(columns=["defaulted"], axis=1)

    # defining target (y) column
    data_y = data["defaulted"]

    # performing train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=t_size, random_state=rndm_state
    )

    return X_train, X_test, y_train, y_test


# creating instances of KMeans
def instantiating_clstrs():
    """
    This function instantiates three KMeans clusters

    Returns: three isntances of KMeans
    """

    # performing k-means clustering on pay_*, bill_amt*, pay_amt*

    # instantiating KMeans
    clustering_pay = KMeans(n_clusters=6, random_state=46)
    clustering_bill = KMeans(n_clusters=3, random_state=46)
    clustering_pay_amt = KMeans(n_clusters=3, random_state=46)

    return clustering_pay, clustering_bill, clustering_pay_amt


# creating pay_amt clusters
def pay_amt_clstrs(data, clustering_pay_amt):
    """
    This function performs KMeans clustering on pay_amt*.
    It fits KMeans model instance on the mentioned columns.
    It then adds those clusters as feature set to the existing
    dataframe given as input.

    Parameters: dataframe, one KMeans instance (object)

    Returns: dataframe
    """

    # filtering all columns starting with "pay_amt"
    filter_pay_amt = data.filter(regex=r"^pay_amt\d", axis=1)

    # fitting K-Means
    clustering_pay_amt.fit(filter_pay_amt)

    # apply the labels
    pay_amt_labels = clustering_pay_amt.labels_

    # creating new feature columns
    data.name = "{}".format(data)
    df_name = data.name + "_clstrs"
    df_name = data.copy()
    df_name["pay_amt_clusters"] = pay_amt_labels

    # one-hot encoding the new columns created using pandas
    df_name = pd.get_dummies(df_name, columns=["pay_amt_clusters"])

    return df_name


# creating pay clusters
def pay_clstrs(data, clustering_pay):
    """
    This function performs KMeans clustering on pay_* and it fits
    KMeans model instance on the mentioned columns.
    It then adds that cluster as feature set to the existing
    dataframe given as input.

    Parameters: dataframe, one KMeans instance (object)

    Returns: dataframe
    """

    # filtering all columns starting with "pay_"
    filter_pay = data.filter(regex=r"^pay_\d{1}_-?\d", axis=1)

    # fitting K-Means
    clustering_pay.fit(filter_pay)

    # apply the labels
    pay_labels = clustering_pay.labels_

    # creating new feature columns
    data.name = "{}".format(data)
    df_name = data.name + "_clstrs"
    df_name = data.copy()
    df_name["pay_clusters"] = pay_labels

    # one-hot encoding the new columns created using pandas
    df_name = pd.get_dummies(df_name, columns=["pay_clusters"])

    return df_name


# creating bill clusters
def bill_clstrs(data, clustering_bill):
    """
    This function performs KMeans clustering on bill_amt*.
    It fits KMeans model instance on the mentioned columns.
    It then adds those clusters as feature set to the existing
    dataframe given as input.

    Parameters: dataframe, one KMeans instance (object)

    Returns: dataframe
    """

    # filtering all columns starting with "bill_amt"
    filter_bill = data.filter(regex=r"^bill_amt\d", axis=1)

    # fitting K-Means
    clustering_bill.fit(filter_bill)

    # apply the labels
    bill_labels = clustering_bill.labels_

    # creating new feature columns
    data.name = "{}".format(data)
    df_name = data.name + "_clstrs"
    df_name = data.copy()
    df_name["bill_clusters"] = bill_labels

    # one-hot encoding the new columns created using pandas
    df_name = pd.get_dummies(df_name, columns=["bill_clusters"])

    return df_name


# applying random forest
def random_forest(X_train, y_train):
    """
    This function fits random forest model on the input dataframes
    and calculates the top 15 features that influence the model
    performance

    Parameters: two dataframes (X_train, y_train)

    Returns: model
    """
    clf = RandomForestClassifier(
        criterion="gini",
        max_depth=6,
        max_features="auto",
        n_estimators=11,
        random_state=46,
    )
    clf.fit(X_train, y_train)

    # studying top 15 important features
    feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(15).plot(kind="barh")
    plt.title("Top 15 important features")
    plt.show()

    return clf


# dropping pay_* since the clusters formed are representing them
def drop_pay(data):
    """
    This function drops the pay_* columns as the clusters formed
    represent them.

    Parameters: dataframe

    Returns: dataframe
    """
    filter_pay = data.filter(regex=r"^pay_\d{1}_-?\d", axis=1)
    data.drop(columns=filter_pay.columns, axis=1, inplace=True)

    return data


# dropping bill_clusters and pay_amt_clusters since the clusters
# formed do not affect the model
def drop_clstrs(data):
    """
    This function drops the bill_clusters and pay_amt_clusters
    columns as the clusters formed do not affect the model.

    Parameters: dataframe

    Returns: dataframe
    """

    # filtering and dropping pay_amt_clusters
    filter_pay_amt_clstrs = data.filter(regex=r"^pay_amt_clusters_\d", axis=1)
    data.drop(columns=filter_pay_amt_clstrs.columns, axis=1, inplace=True)

    # filtering and dropping bill_clusters
    filter_bill_clstrs = data.filter(regex=r"^bill_clusters_\d", axis=1)
    data.drop(columns=filter_bill_clstrs.columns, axis=1, inplace=True)

    return data


# evaluation metric
def scoring(y_test, X_test, clf):
    """
    This function predicts on X_test and returns the
    classification report

    Parameters: two dataframes (y_test, X_test)and fitted model

    Returns: object explaining classification report
    """
    # store the predicted values of test data
    y_pred = clf.predict(X_test)

    eval_report = classification_report(y_test, y_pred)

    return eval_report
