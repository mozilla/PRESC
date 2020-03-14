import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score


def Tuning_LogiticRegression(X, y):
    """ Tuning Logistic Regression to increase the accuracy of model """

    parameters = {
        "penalty": ["l1", "l2"],
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    LR_grid = GridSearchCV(LogisticRegression(), param_grid=parameters, cv=5)
    LR_grid = LR_grid.fit(X, y)
    classifier = LogisticRegression(
              penalty=LR_grid.best_estimator_.get_params()["penalty"],
              C=LR_grid.best_estimator_.get_params()["C"],
              )
    return classifier.fit(X, y)

def SVM_train(X, y):

    """ SVM Classifier"""
    # kernel =' poly ' is taking infinte time that's why it is not added.
    params_grid = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]}]

    svm_grid = GridSearchCV(SVC(), params_grid, cv=5)
    svm_grid = svm_grid.fit(X, y)
    classifier = SVC(
        kernel=svm_grid.best_estimator_.kernel,
        C=svm_grid.best_estimator_.C,
        gamma=svm_grid.best_estimator_.gamma,
    )

    return classifier.fit(X, y)