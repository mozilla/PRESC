import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn.svm import SVC
import Visualization_for_misclassifications as visual


def data_stats(dataset):
    """ Shows some basic stats of the dataset"""
    print("=========== SOME STATS of Dataset ===========")
    print("Shape of the dataset: " + str(dataset.shape) + "\n")
    print("List of attribute columns", list(dataset.columns))
    print("\n")
    list_cat = dataset.Class.unique()
    print("List of Categories ", list_cat, "\n")


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


def train_data_test_data_split(dataset):
    X = dataset.iloc[:, 0:-1].values
    y = dataset.iloc[:, -1].values
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
        X, y, test_size=0.2, random_state=21
    )
    return X_train, X_test, y_train, y_test


def train_randomforest(X_train, y_train):
    """ training model on train data"""
    classifier = RandomForestClassifier(
        n_estimators=10, criterion="entropy", random_state=42
    )
    classifier.fit(X_train, y_train)
    return classifier


def train_SVM(X_train, y_train):
    params_grid = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]}]
    svm_model = GridSearchCV(SVC(), params_grid, cv=5)
    svm_model.fit(X_train, y_train)
    svclassifier = SVC(
        kernel=svm_model.best_estimator_.kernel,
        C=svm_model.best_estimator_.C,
        gamma=svm_model.best_estimator_.gamma,
    )
    svclassifier.fit(X_train, y_train)
    return svclassifier


def test(classifier, X_test):
    """ testing model on test data"""
    y_pred = classifier.predict(X_test)
    return y_pred


def untokenizing_testdata(y_test, definitions):
    """Converting numeric target values back to original labels"""
    reversefactor = dict(zip(range(len(definitions) + 1), definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    return y_test


def untokenizing_prediction(y_pred, definitions):
    """Converting numeric predicted values back to original labels"""
    reversefactor = dict(zip(range(len(definitions) + 1), definitions))
    y_pred = np.vectorize(reversefactor.get)(y_pred)
    return y_pred


def create_confusion_matrix(y_test, y_pred):
    """ Creates Cinfusion Matrix and summary of evaluation metric """

    labels = ["van", "saab", "bus", "opel"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    return df_cm


def display_confusion_matrix(df_cm):
    """ displays confusion matrix of dataframe provided)"""

    sn.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel("Real Vehicle Category")
    plt.ylabel("Predicted Vehicle Category")
    print("====================== Confusion Matrix=====================")


def display_classification_report(y_test, y_pred):
    print("============== Summary of all evaluation metics ===============")
    print(classification_report(y_test, y_pred))


def model_evaluation(models, X_train, y_train):
    """ Checking accuaracy of different models and plotting it for comparison"""
    print(
        "Evaluating performance of various classifier:\n ==================================== \n Random Forest Classifier, K Neighbor Classifier, RBF SVM, Naive Bayes, Logistic Regression, Decision Tree\n "
    )

    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, X_train, y_train, scoring="accuracy", cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy"])
    model_evaluation_plot(cv_df)
    cv_df.groupby("model_name").accuracy.mean()


def model_evaluation_plot(cv_df):
    """ Display dataframe containing model and their accuracy for comparison"""
    figure(num=None, figsize=(12, 12), dpi=80, facecolor="w", edgecolor="k")
    sns.boxplot(x="model_name", y="accuracy", data=cv_df)
    sns.stripplot(
        x="model_name",
        y="accuracy",
        data=cv_df,
        size=8,
        jitter=True,
        edgecolor="gray",
        linewidth=2,
    )
    plt.show()


def model_false_positive_visualisation(
    X_train, X_test, y_train, y_test, models, definitions
):
    """Visualises all the missclassified datapoints in each class.
    parameters:
    X_train: Training datapoints, 
    X_test: Testing datapoints, 
    y_train: Training labels, 
    y_test: Testing Labels,
    models: list  of classifiers to use"""
    y_test = untokenizing_testdata(y_test, definitions)

    for model in models:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        y_pred = test(model, X_test)
        print(
            "================= Graph of False Positives for ",
            model_name,
            "==============",
        )
        y_pred = untokenizing_prediction(y_pred, definitions)
        visual.missclassified_data_category_frquency(y_test, y_pred)
