""" This file contain evaluation metrics for the model"""

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix, plot_confusion_matrix,average_precision_score,precision_recall_curve, plot_precision_recall_curve, f1_score, precision_score, recall_score
import pandas as pd

from sklearn.utils import class_weight


def evaluate(classifier, X_test, y_test):
    y_score = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_score)
    f_score = f1_score(y_test, y_score)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score)

    # print("Accuracy : ",accuracy)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 score : ",f_score)

    return accuracy, precision, recall, f_score

