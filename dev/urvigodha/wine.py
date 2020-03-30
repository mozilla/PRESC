import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def transform(X_train, X_test):
    from sklearn.preprocessing import PowerTransformer

    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    transformer.fit(X_train)
    X_train_array = transformer.transform(X_train)
    X_train = pd.DataFrame(
        data=X_train_array, index=X_train.index, columns=X_train.columns
    )
    X_test_array = transformer.transform(X_test)
    X_test = pd.DataFrame(data=X_test_array, index=X_test.index, columns=X_test.columns)
    return X_train, X_test


def scale(X_train, X_test):
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_array = scaler.transform(X_train)
    X_train = pd.DataFrame(
        data=X_train_array, index=X_train.index, columns=X_train.columns
    )
    X_test_array = scaler.transform(X_test)
    X_test = pd.DataFrame(data=X_test_array, index=X_test.index, columns=X_test.columns)
    return X_train, X_test


def model(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix

    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    y_test_pred = RF.predict(X_test)
    print("Train accuracy: ", RF.score(X_train, y_train))
    print("Test accuracy: ", RF.score(X_test, y_test))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_test_pred))
