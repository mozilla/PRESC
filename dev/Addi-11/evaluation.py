# This file contain evaluation metrics for the model

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd

from sklearn.utils import class_weight

def evaluate(classifier, x_val, y_val):
    """
    This function predicts the values for a trained model and calculates the evaluation metrics accuracy, precision, recall, f_score on the validation set. 
    Parameters :
        classifier : trained classification model
        x_val : array-like, shape( n_samples, n_features)
        y_val : array-like, of length n_samples

    Returns :
        accuracy : float
        precision : float
        recall : float
        f_score : float
        y_score : array-like of length n_features, 
    """
    y_score = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, y_score)
    f_score = f1_score(y_val, y_score)
    precision = precision_score(y_val, y_score)
    recall = recall_score(y_val, y_score)

    return accuracy, precision, recall, f_score, y_score
