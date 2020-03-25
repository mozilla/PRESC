import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.stats import gmean

def read_csv():
	filePath = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"), "winequality.csv")
	wineQuality = pd.read_csv(filePath)
	return wineQuality

def is_null_values(myTest):
    return myTest.isnull().sum()


def drop_quality_column(dtf, column_name):
    dtf = dtf.drop([column_name], axis=1)
    return dtf


def extract_features(dataset):
    features = dataset.iloc[:, 0:11]
    return features


def label_encoding(dataframe_column):
    labels = dataframe_column
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    return encoded_labels


def class_distribution(df):
    print(df.groupby("recommend").size())


def data_distribution(data_features):
    for feature in data_features:
        f, axes = plt.subplots(figsize=(14, 4))
        sns.distplot(data_features[feature], color="m")
        axes.set_xlabel(feature, fontsize=18)
        axes.set_ylabel("Count", fontsize=18)


def feature_correlations(data_features):
    feature_corr = data_features.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(feature_corr, cmap="coolwarm", annot=True)


def bivar_plot(dtf, v1, v2):
    sns.pairplot(dtf, vars=[v1, v2])
    plt.show()


def feature_scaling(data_features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_features)
    return scaled_features


def create_ml_models(ml_models):
    ml_models.append(("k Nearest Neighbors", KNeighborsClassifier()))
    ml_models.append(("Decision Tree", DecisionTreeClassifier()))


def cross_validation_score(ml_models, feature_train, label_train, num_folds):
    model_names = []
    model_accuracies = []
    for ml_name, ml_model in ml_models:
        skfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)
        cv_results = cross_val_score(
            ml_model, feature_train, label_train, cv=skfold, scoring="accuracy"
        )
        model_names.append(ml_name)
        model_accuracies.append(cv_results.mean())
        cv_results = cv_results.mean() * 100
        print(ml_name, round(cv_results, 2))


def traversal_space_cross_val(ml_models, feature_train, label_train):
    number_of_folds = np.arange(2, 11)
    for fold_num in number_of_folds:
        print("Estimated Accuracy for " + str(fold_num) + " number of folds")
        cross_validation_score(ml_models, feature_train, label_train, fold_num)
        print("")


def grid_search_knn(feature_train, label_train):
    knn = KNeighborsClassifier()
    param_grid = {"n_neighbors": np.arange(1, 14, 1)}
    grid_search = GridSearchCV(knn, param_grid, cv=10)
    grid_search.fit(feature_train, label_train)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_

    return best_estimator, round(best_score, 2)


def grid_search_dt(feature_train, label_train):
    model_dc = DecisionTreeClassifier()
    param_grid = {"criterion": ["gini", "entropy"], "max_depth": np.arange(4, 16, 1)}
    grid_search = GridSearchCV(model_dc, param_grid, cv=10)
    grid_search.fit(feature_train, label_train)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_estimator, round(best_score, 2)


def grid_search_svm(feature_train, label_train):
    model_svm = SVC()
    param_grid = {
        "kernel": ["linear", "rbf", "poly"],
        "degree": [2, 3, 4],
        "C": [1, 10, 50, 100],
    }
    grid_search = GridSearchCV(model_svm, param_grid, cv=10, n_jobs=-1)
    grid_search.fit(feature_train, label_train)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_estimator, round(best_score, 2)


def knn_eval_model_predictions(feature_train, label_train, feature_test, label_test):
    knn_model = KNeighborsClassifier(
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        n_neighbors=1,
        p=2,
        weights="uniform",
    )
    knn_model.fit(feature_train, label_train)
    knn_pred = knn_model.predict(feature_test)
    print(
        "Accuracy of kNN:", round(np.sum(knn_pred == label_test) / len(label_test), 2)
    )
    print("\n")
    print("Confusion Matrix for kNN")
    print(confusion_matrix(label_test, knn_pred))
    print("\n")
    print("Classification Report for kNN")
    print(classification_report(label_test, knn_pred))


def dc_eval_model_predictions(feature_train, label_train, feature_test, label_test):
    dc_model = DecisionTreeClassifier(
        class_weight=None,
        criterion="entropy",
        max_depth=10,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        presort=False,
        random_state=None,
        splitter="best",
    )
    dc_model.fit(feature_train, label_train)
    dc_pred = dc_model.predict(feature_test)
    print(
        "Accuracy of Decision Tree:",
        round(np.sum(dc_pred == label_test) / len(label_test), 2),
    )
    print("Confusion Matrix for Decision Tree")
    print(confusion_matrix(label_test, dc_pred))
    print("Classification Report for Decision Tree")
    print(classification_report(label_test, dc_pred))
