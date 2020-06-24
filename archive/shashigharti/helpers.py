from sklearn.svm import SVC  # SVM Classifier
from sklearn.model_selection import train_test_split
from sklearn import (
    metrics,
)  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd


def fix_outlier_with_mean(df, df_columns_with_outliers):
    """ This function takes data as input and replaces outliers of each columns by mean
    
    Args:
        df: dataframe of all data
    Returns:
        data_df: processed dataframe
        
    """
    data_df = df.copy()

    # Fill null
    data_df.fillna(data_df.mean(), inplace=True)

    # Replace outliers with mean value. Mean is calculated with data rows excluding outliers
    for i, column in enumerate(df_columns_with_outliers.columns):
        data_df.loc[
            data_df[column] > df_columns_with_outliers[column][0], column
        ] = data_df[data_df[column] <= df_columns_with_outliers[column][0]][
            column
        ].mean()

    data_df["Class"] = pd.Categorical(data_df["Class"]).codes

    return data_df


def fix_outlier_with_boundary_value(df, df_columns_with_outliers):
    """ This function takes data as input and replaces outliers > max value of each columns by max value
    
    Args:
        df: dataframe of all data
    Returns:
        data_df: processed dataframe
        
    """
    data_df = df.copy()

    # Fill null
    data_df.fillna(data_df.mean(), inplace=True)

    # Replace outliers with max boundary value
    for i, column in enumerate(df_columns_with_outliers.columns):
        data_df.loc[
            data_df[column] > df_columns_with_outliers[column][0], column
        ] = df_columns_with_outliers[column][0]

    data_df["Class"] = pd.Categorical(data_df["Class"]).codes

    return data_df


def remove_outliers(df, df_columns_with_outliers):
    """ This function takes data as input and removes the outliers
    
    Args:
        df: dataframe of all data
    Returns:
        data_df: processed dataframe
        
    """
    data_df = df.copy()

    # Fill null
    data_df.fillna(data_df.mean(), inplace=True)

    # Remove outliers based on max value identified earlier from boxplot
    for i, column in enumerate(df_columns_with_outliers.columns):
        data_df = data_df[data_df[column] < df_columns_with_outliers[column][0]]

    data_df["Class"] = pd.Categorical(data_df["Class"]).codes

    # reset the index post cleaning the outliers
    data_df = data_df.reset_index(drop=True)

    return data_df


def train_svm(X, y, scaler=None, ratio=0.4):
    """ This function takes train data, labels as input and trains the SVM model.
    
    Args:
        X: Training Data(features columns).
        y: Labels for each row
        scaler: Transformation class object
        ratio: (optional) Split ratio for train and test data

    Returns:
        tuple: y_test, y_pred values
        
    """

    ## Split the train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=1
    )

    if scaler is not None:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        print(type(scaler))

    ## Build the SVM model on training data
    model_svc = SVC(gamma="auto")
    model_svc.fit(X_train, y_train)
    y_pred = model_svc.predict(X_test)

    return y_test, y_pred


def train_svm_with_hyperparameter_tuning(X, y, param_grid, scaler=None, ratio=0.4):
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

    ## Split the train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=1
    )

    if scaler is not None:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        print(type(scaler))

    svm_grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    svm_grid.fit(X_train, y_train)
    y_pred = svm_grid.predict(X_test)

    return y_test, y_pred


def print_results(y_test, y_pred):
    """ This function takes y_test and y_pred values and print the results
    
    Args:
        y_test: Training Data(features columns).
        y_pred: Labels for each row

    Returns:
        None

    """

    # Print Model Accuracy
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
