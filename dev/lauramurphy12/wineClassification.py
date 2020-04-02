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
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
from scipy.stats import gmean
from sklearn.metrics import fbeta_score, make_scorer


def read_csv():
    filePath = os.path.join(
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "datasets",
        ),
        "winequality.csv",
    )
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


def class_distribution(target_labels):
    target_var = Counter(target_labels)
    for class_label, class_count in target_var.items():
        class_distribution = (class_count / len(target_labels)) * 100
        print(
            "Class Label=%s, Class Count=%d, Distribution=%.2f%%"
            % (class_label, class_count, class_distribution)
        )


def data_distribution(data_features):
    for feature in data_features:
        f, axes = plt.subplots(figsize=(14, 4))
        sns.distplot(data_features[feature], color="m")
        axes.set_xlabel(feature, fontsize=18)
        axes.set_ylabel("Count", fontsize=18)


def feature_correlations(dtf):
    feature_corr = dtf.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(feature_corr, cmap="coolwarm", annot=True)


def bivar_plot(dtf, v1, v2):
    sns.pairplot(dtf, vars=[v1, v2])
    plt.show()


def feature_scaling(data_features, feature_column_names):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_features)
    scaled_f = pd.DataFrame(data=scaled_features)
    scaled_f.columns = feature_column_names
    return scaled_f


def choose_best_features(feature_train, label_train, feature_column_names):
    kbest_features = SelectKBest(score_func=f_classif, k=4)
    kbest_features.fit(feature_train, label_train)
    best_scores = kbest_features.scores_
    feature_scores_df = pd.DataFrame(kbest_features.scores_)
    feature_names_df = pd.DataFrame(feature_column_names)
    top_features = pd.concat([feature_names_df, feature_scores_df], axis=1)
    top_features.columns = ["Wine Features", "Feature Scores"]
    return top_features


def create_ml_models(ml_models):
    ml_models.append(("k Nearest Neighbors", KNeighborsClassifier()))
    ml_models.append(("Decision Tree", DecisionTreeClassifier()))


def evaluate_models_trainTestSplit(features, labels, ml_models):
    ts = np.arange(0.2, 1.0, 0.05)
    traverse_splits = pd.DataFrame(
        columns=["Model", "Test Size", "Train Size", "Accuracy", "Precision"]
    )
    for ts_size in ts:
        tr_size = 1 - ts_size
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=ts_size, random_state=42
        )
        for ml_name, ml_model in ml_models:
            ml_model.fit(X_train, y_train)
            ml_pred = ml_model.predict(X_test)
            ml_accuracy = accuracy_score(y_test, ml_pred)
            precision = precision_score(y_test, ml_pred)
            traverse_splits = traverse_splits.append(
                {
                    "Model": ml_name,
                    "Test Size": ts_size,
                    "Train Size": tr_size,
                    "Accuracy": ml_accuracy,
                    "Precision": precision,
                },
                ignore_index=True,
            )
    return traverse_splits


def evaluate_ml_model_crossVal(features, labels, ml_models, fold_num):
    k_folds = np.arange(2, fold_num)
    traverse_folds = pd.DataFrame(
        columns=["Model", "Number of Folds", "Accuracy", "Precision", "fbeta"]
    )
    model_names = []
    model_accuracies = []
    model_precision = []
    model_fbeta = []
    for fold in k_folds:
        for ml_name, ml_model in ml_models:
            repeated_skFold = RepeatedStratifiedKFold(
                n_splits=fold, n_repeats=2, random_state=42
            )
            fbeta_metric = make_scorer(fbeta_score, beta=0.5)
            precision_metric = make_scorer(precision_score)
            accuracy_scores = cross_val_score(
                ml_model, features, labels, scoring="accuracy", cv=repeated_skFold
            )
            precision_scores = cross_val_score(
                ml_model, features, labels, scoring=precision_metric, cv=repeated_skFold
            )
            fbeta_scores = cross_val_score(
                ml_model, features, labels, scoring=fbeta_metric, cv=repeated_skFold
            )
            model_names.append(ml_name)
            model_accuracies.append(accuracy_scores.mean())
            model_precision.append(precision_scores.mean())
            model_precision.append(fbeta_scores.mean())
            traverse_folds = traverse_folds.append(
                {
                    "Model": ml_name,
                    "Number of Folds": fold,
                    "Accuracy": accuracy_scores.mean(),
                    "Precision": precision_scores.mean(),
                    "fbeta": fbeta_scores.mean(),
                },
                ignore_index=True,
            )
    return traverse_folds


def traversal_space_cross_val(ml_models, feature, label):
    number_of_folds = np.arange(2, 11)
    for fold_num in number_of_folds:
        print("Estimated Accuracy for " + str(fold_num) + " number of folds")
        evaluate_ml_models(ml_models, feature, label, fold_num)
        print("")


def grid_search_knn(features, labels):
    knn = KNeighborsClassifier()
    param_grid = {"n_neighbors": np.arange(1, 14, 1)}
    grid_search = GridSearchCV(knn, param_grid, cv=10)
    grid_search.fit(features, labels)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_

    return best_estimator, round(best_score, 2)


def grid_search_dt(features, labels):
    model_dc = DecisionTreeClassifier()
    param_grid = {"criterion": ["gini", "entropy"], "max_depth": np.arange(4, 16, 1)}
    grid_search = GridSearchCV(model_dc, param_grid, cv=10)
    grid_search.fit(features, labels)
    best_estimator = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_estimator, round(best_score, 4)


def knn_eval_model_predictions(features, labels):
    knn_model = KNeighborsClassifier(
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        n_neighbors=12,
        p=2,
        weights="uniform",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42
    )
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    print("Accuracy of kNN:", accuracy_score(y_test, knn_pred))
    print("\n")
    print("Confusion Matrix for kNN")
    print(confusion_matrix(y_test, knn_pred))
    print("\n")
    print("Classification Report for kNN")
    print(classification_report(y_test, knn_pred))


def dc_eval_model_predictions(features, labels):
    dc_model = DecisionTreeClassifier(
        class_weight=None,
        criterion="entropy",
        max_depth=6,
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
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42
    )
    dc_model.fit(X_train, y_train)
    dc_pred = dc_model.predict(X_test)
    print("Accuracy of Decision Tree:", accuracy_score(y_test, dc_pred))
    print("Confusion Matrix for Decision Tree")
    print(confusion_matrix(y_test, dc_pred))
    print("Classification Report for Decision Tree")
    print(classification_report(y_test, dc_pred))
