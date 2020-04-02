import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from IPython.display import display


def read_data(path=None):
    """Function to read dataset from file"""
    if path is None:
        print("Path not specified.")
        return None, None, None, None, None
    elif os.path.exists(path) is False:
        print("File not found.")
        return None, None, None, None, None
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    data = data.to_numpy()

    # Data processing specifically for vehicles.csv
    X = data[:, :18]
    y = data[:, -1]

    classes = np.unique(y)
    indices = list(np.arange(len(classes)))
    class_map = dict(list(zip(classes, indices)))
    y = np.array([class_map[y[i]] for i in range(len(y))])

    display(pd.DataFrame({"Mapping": list(class_map.keys())}))

    return X, y, class_map, data, df


def dim_reduction(method="pca", X=None, y=None, k=2):
    """Function to compute PCA/LDA of data points to k dimensions"""
    if method.lower() == "pca":
        if X is None:
            return None
        model = PCA(n_components=k)
    elif method.lower() == "lda":
        if X is None or y is None:
            return None
        model = LDA(n_components=k)
    projection = model.fit_transform(X, y)
    return projection


def show_scree(X=None, n=None):
    """Function to display Scree plot of data"""
    if X is None:
        return

    pca = PCA()
    pca.fit(X)
    eigenvalue_ratio = pca.explained_variance_ratio_

    if n is None:
        n = len(eigenvalue_ratio)

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(n) + 1, eigenvalue_ratio[:n], "o-")
    plt.xlabel("Principal Components")
    plt.ylabel("Proportion of Variance explained")
    plt.title("Scree Plot")
    plt.show()

def convert_binary(y=None, group=None, labels=None):
    """Function to convert the class labels to binary based on specified groupings"""
    if y is None:
        print('Data not provided.')
        return None 
    elif group is None:
        print('Grouping not specified.')
        return None

    if labels is None:
        labels = [-1, 1]

    y[np.where(np.isin(y, group[0]))] = labels[0]
    y[np.where(np.isin(y, group[1]))] = labels[1]

    return y


def split_data(X=None, y=None, train_size=None):
    """Function to get train-test split of the dataset"""
    if X is None or y is None:
        return None, None, None, None
    elif train_size is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    return X_train, X_test, y_train, y_test


def visualize_data(method="LDA", X=None, y=None, n=None):
    """Function to visualize data using LDA or PCA in 3-D"""
    if X is None or y is None:
        return False
    else:
        projection = dim_reduction(method.lower(), X, y, 2)

    show_scree(X, n)

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(projection[:, 0], projection[:, 1], c=y)
    plt.title(method)
    plt.show()

    return True
