import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pandas import set_option
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def loadDataAndPrintHead(file_path):
    """
    Loads data using pandas read_csv.
    Args: Local path to the dataset file as a string
    """
    df = pd.read_csv(file_path)
    print(df.head())
    return df


def checkNullValues(df):
    """
    Checks if any of the columns has null values.
    """
    print(df.isnull().sum())


def makeFeaturesAndLabel(df):
    """
    Drops the column 'id' and makes label and features.
    Args: The main dataframe
    Returns:
    y --> label
    featurs --> set of features
    """
    df.drop("id", axis=1, inplace=True)
    y = df["defaulted"]
    features = df.drop(["defaulted"], axis=1)
    return y, features


def groupUnknownCategories(df):
    """
    Group the categories of columns that are not known.
    'education' - 1,2,3 are known, group rest in 'others' ie 4
    'marriage' - 1,2 are known, grous rest in 'others' ie 3
    Args: the dataframe
    """
    df["education"] = np.where((df["education"] == 5), 4, df["education"])
    df["education"] = np.where(df["education"] == 6, 4, df["education"])
    df["education"] = np.where(df["education"] == 0, 4, df["education"])
    df["marriage"] = np.where(df["marriage"] == 0, 3, df["marriage"])


def plotGrapthForDefaults(y):
    """
    Plots a graph to show default values
    Agrs: The label y
    """

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    index, counts = np.unique(y, return_counts=True)
    ax.bar(index, counts)
    ax.set_xlabel("default payment next month")
    ax.set_ylabel("Number of data points")
    plt.title("Number of default payments", fontsize=15)
    plt.show()


def plotGrapthForAttributes(df):
    """
    Plots Graphs for each attibute for EDA
    Args: The dataframe
    """
    # Creating a new dataframe with categorical variables
    subset = df[
        [
            "sex",
            "education",
            "marriage",
            "pay_0",
            "pay_2",
            "pay_3",
            "pay_4",
            "pay_5",
            "pay_6",
            "defaulted",
        ]
    ]

    f, axes = plt.subplots(3, 3, figsize=(20, 15), facecolor="white")
    f.suptitle("FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)")
    ax1 = sns.countplot(
        x="sex", hue="defaulted", data=subset, palette="Blues", ax=axes[0, 0]
    )
    ax2 = sns.countplot(
        x="education", hue="defaulted", data=subset, palette="Blues", ax=axes[0, 1]
    )
    ax3 = sns.countplot(
        x="marriage", hue="defaulted", data=subset, palette="Blues", ax=axes[0, 2]
    )
    ax4 = sns.countplot(
        x="pay_0", hue="defaulted", data=subset, palette="Blues", ax=axes[1, 0]
    )
    ax5 = sns.countplot(
        x="pay_2", hue="defaulted", data=subset, palette="Blues", ax=axes[1, 1]
    )
    ax6 = sns.countplot(
        x="pay_3", hue="defaulted", data=subset, palette="Blues", ax=axes[1, 2]
    )
    ax7 = sns.countplot(
        x="pay_4", hue="defaulted", data=subset, palette="Blues", ax=axes[2, 0]
    )
    ax8 = sns.countplot(
        x="pay_5", hue="defaulted", data=subset, palette="Blues", ax=axes[2, 1]
    )
    ax9 = sns.countplot(
        x="pay_6", hue="defaulted", data=subset, palette="Blues", ax=axes[2, 2]
    )


def splitData(df, features, y):
    """
    Splits the data into test and train
    Args: The main dataframe, label y and the features
    Returns: X_train, X_test, y_train, y_test the training and testing datasets respectively
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.30, random_state=42
    )
    # X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.67, random_state=42)
    # X_cv.shape
    return X_train, X_test, y_train, y_test


def standardize(x, x_train):
    """
    Utility function for standardizeData()
    standardizes according to train_data mean and std
    Args: x - data to be standardize
          x_train - trainind data
    Returns: standardized Data x
    """
    x = x.astype(float)
    x_train = x_train.astype(float)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x = scaler.transform(x)
    return x


def standardizeData(X_test, X_train):
    """
    Uses standardize to normalize data.
    Args : x - data to be standardized
    """
    col_to_norm = [
        "limit_bal",
        "age",
        "bill_amt1",
        "bill_amt2",
        "bill_amt3",
        "bill_amt4",
        "bill_amt5",
        "bill_amt6",
        "pay_amt1",
        "pay_amt2",
        "pay_amt3",
        "pay_amt4",
        "pay_amt5",
        "pay_amt6",
    ]

    X_test[col_to_norm] = standardize(X_test[col_to_norm], X_train[col_to_norm])
    # X_cv[col_to_norm] = standardize(X_cv[col_to_norm], X_train[col_to_norm])
    X_train[col_to_norm] = standardize(X_train[col_to_norm], X_train[col_to_norm])
    print(X_train.head())


def logisticRegressionModel(X_train, y_train, X_test, y_test):
    """
    Fits the training data in a sklearn LogisticRegression model.
    Args: X_train, y_train, X_test, y_test
    Returns: preds - The predictions for the X_test
             score - Accuracy for the model on test data
    """
    clfModel = LogisticRegression(random_state=0).fit(
        X_train, y_train
    )  # fitting the model on train data
    preds = clfModel.predict(X_test)  # predictions
    score = clfModel.score(X_test, y_test)  # accuracy
    return preds, score


def svmModel(X_train, y_train, X_test, y_test):
    """
    Fits the training data in a sklearn.svm.SVC model.
    Args: X_train, y_train, X_test, y_test
    Returns: preds - The predictions for the X_test
             score - Accuracy for the model on test data
    """
    SVM_Model = SVC(gamma="auto", verbose=True)
    SVM_Model.fit(X_train, y_train)  # fitting on train data
    score = f"Accuracy - : {SVM_Model.score(X_test, y_test):.3f}"
    prediction_SVM = SVM_Model.predict(X_test)
    return score, prediction_SVM
