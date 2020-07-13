# This file evaluates  the importance of a datapoint in the training dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from classifiers import Classifier

# =============== METHOD : 1 =============

# the point misclassified by all classifiers : outliers
# the point misclassified by some and not by others : has imp. info
# the point correctly classified by all : easy point

def cal_point_type(x_train, x_val, y_train, y_val):
    """
    Calculates the misclassified values of each classifier

    Parameters:
        x_train: array:like, shape(n_train_samples, n_features)
        x_val: array:like, shape(n_val_samples, n_features)
        y_train: of length n_train_samples
        y_val: of length n_val_samples

    Returns:
        outliers: array:like
        point_score: dictionary
        easy_points: array:like
    """
    # list of all classifiers
    classifiers = [
        "KNeighbors",
        "Random_Forest",
        "svm_classifier",
        "Gaussian",
        "Decision_Tree",
        "Logistic_Reg",
    ]
    model = Classifier()

    # create dictionay to store misclassified values for each classifier
    err_indexes = {}
    correct_indexes = {}
    wt = {}

    for clf in classifiers:
        train_clf = getattr(model, clf)(x_train, y_train)
        y_score = train_clf.predict(x_val)
        # get the indexes of misclassified values
        err_indexes[clf] = np.where(y_score != y_val)
        correct_indexes[clf] = np.where(y_score == y_val)
        # associate wt to each model, based on its accuracy
        acc = accuracy_score(y_val, y_score)
        wt[clf] = 1 / (1 - np.power(acc, 2))

    # calculating outliers
    outliers = err_indexes["KNeighbors"]

    for clf in classifiers:
        outliers = np.intersect1d(outliers, err_indexes[clf])

    # calculating points with trivial info : the misclassified points by model with high accuracy are a subset of misclassified points by models with less accuracy.
    # print('Points associated with each model :', wt)

    # correctly by k:nn but not by random:forest
    s1 = wt["KNeighbors"] - wt["Random_Forest"]
    pt1 = np.intersect1d(correct_indexes["KNeighbors"], err_indexes["Random_Forest"])

    # correctly by random:forest not by decision:tree
    s2 = wt["Random_Forest"] - wt["Decision_Tree"]
    pt2 = np.intersect1d(correct_indexes["Random_Forest"], err_indexes["Decision_Tree"])

    # correctly by decision:tree not by logistic regression
    s3 = wt["Decision_Tree"] - wt["Logistic_Reg"]
    pt3 = np.intersect1d(correct_indexes["Decision_Tree"], err_indexes["Logistic_Reg"])

    # correctly by logistic regression not by Gaussian
    s4 = wt["Logistic_Reg"] - wt["Gaussian"]
    pt4 = np.intersect1d(correct_indexes["Logistic_Reg"], err_indexes["Gaussian"])

    # correctly by Gaussian not by svm
    s5 = wt["Gaussian"] - wt["svm_classifier"]
    pt5 = np.intersect1d(correct_indexes["Gaussian"], err_indexes["svm_classifier"])

    point_score = {
        "p1": (s1, pt1),
        "p2": (s2, pt2),
        "p3": (s3, pt3),
        "p4": (s4, pt4),
        "p5": (s5, pt5),
    }

    # calculating easy:points
    easy_points = correct_indexes["KNeighbors"]
    for clf in classifiers:
        easy_points = np.intersect1d(easy_points, correct_indexes[clf])

    return outliers, point_score, easy_points

def display_point_type(outliers, point_score, easy_points):
    """
    Plots a bar graph to count the values of each point type.

    Parameters:
        x_train: array:like, shape(n_train_samples, n_features)
        x_val: array:like, shape(n_val_samples, n_features)
        y_train: of length n_train_samples
        y_val: of length n_val_samples

    """

    labels = ("Outliers", "p1", "p2", "p3", "p4", "p5", "Easy:Points")
    x_pos = np.arange(len(labels))
    counts = [
        np.count_nonzero(outliers),
        np.count_nonzero(point_score["p1"][1]),
        np.count_nonzero(point_score["p2"][1]),
        np.count_nonzero(point_score["p3"][1]),
        np.count_nonzero(point_score["p4"][1]),
        np.count_nonzero(point_score["p5"][1]),
        np.count_nonzero(easy_points),
    ]

    plt.bar(x_pos, counts, align="center", alpha=0.5)
    plt.xticks(x_pos, labels)
    plt.ylabel("Counts")
    plt.title("Point Types Values")
    plt.show()

def show_point_types(outliers, point_score, easy_points, val=10):

    print("The first {} values of each point type is displayed.".format(val))

    print("\nOutliers : \n", outliers[:val], "...")
    print(
        "\nP1 score {:.2f} : \n".format(point_score["p1"][0]),
        point_score["p1"][1][:val],
        "...",
    )
    print(
        "\nP2 score {:.2f} : \n".format(point_score["p2"][0]),
        point_score["p2"][1][:val],
        "...",
    )
    print(
        "\nP3 score {:.2f} : \n".format(point_score["p3"][0]),
        point_score["p3"][1][:val],
        "...",
    )
    print(
        "\nP4 score {:.2f} : \n".format(point_score["p4"][0]),
        point_score["p4"][1][:val],
        "...",
    )
    print(
        "\nP5 score {:.2f} : \n".format(point_score["p5"][0]),
        point_score["p5"][1][:val],
        "...",
    )
    print(
        "\nEasy Points(Easily classified by all classifiers) : \n",
        easy_points[:val],
        "...",
    )

def detect_point_type(pt, x_train, outliers, point_score, easy_points):

    if pt > x_train.shape[0] - 1:
        return "Point not in training data"
    if pt in outliers:
        return "Outlier"
    if pt in point_score["p1"][1]:
        return "p1 score {:.2f}".format(point_score["p1"][0])
    if pt in point_score["p2"][1]:
        return "p2 score {:.2f}".format(point_score["p2"][0])
    if pt in point_score["p3"][1]:
        return "p3 score {:.2f}".format(point_score["p3"][0])
    if pt in point_score["p4"][1]:
        return "p4 score {:.2f}".format(point_score["p4"][0])
    if pt in point_score["p5"][1]:
        return "p5 score {:.2f}".format(point_score["p5"][0])
    if pt in easy_points:
        return "Easy Point"
    else:
        return "Avg Point"
