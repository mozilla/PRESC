import pandas as pd
from sklearn.model_selection import train_test_split


def tokenize_target_column(dataset):
    """ tokenize the Class column values to numeric data"""
    factor = pd.factorize(dataset["Class"])
    dataset.Class = factor[0]
    definitions = factor[1]
    print("Updated tokenize 'Class' column - first 5 values")
    print(dataset.Class.head())
    print("Distinct Tokens used for converting Class column to integers")
    print(definitions)
    return definitions


def train_data_test_data_split(X, y, test_size=0.2):
    """ splitting test and training data in 80/20 split"""

    #     print(X[0])
    #     print(y[0])
    #     print(X.shape)
    #     print(y.shape)
    #     print('the data attributes columns')
    #     print(X[:5,:])
    #     print('The target variable: ')
    #     print(y[:5])
    #     Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=21
    )
    return X_train, X_test, y_train, y_test


def training_data_and_target_Label_split(dataset):
    """ return last column of dataset as target y with training dataset  as X. """
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values
    return X, y
