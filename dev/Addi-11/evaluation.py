# This file contain evaluation metrics for the model

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix, plot_confusion_matrix,average_precision_score,precision_recall_curve, plot_precision_recall_curve, f1_score, precision_score, recall_score
import pandas as pd

from sklearn.utils import class_weight


def evaluate(classifier, X_test, y_test):
    '''
    This function predicts the y_values for a trained model and calculates the evaluation metrics accuracy, precision, recall, f_score. 
    Parameters :
        classifier : trained classification model
        X_test : array-like, shape( n_samples, n_features)
        y_test : array-like, of length n_samples

    Returns :
        accuracy : float
        precision : float
        recall : float
        f_score : float
        y_score : array-like of length n_features, 
    '''
    y_score = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_score)
    f_score = f1_score(y_test, y_score)
    precision = precision_score(y_test, y_score)
    recall = recall_score(y_test, y_score, zero_division=1)

    return accuracy, precision, recall, f_score, y_score

