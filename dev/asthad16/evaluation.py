#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


# In[ ]:


def acc_score(y_test, y_pred):
    """it is returning the rstio of number of correct predictions to the total number of predictions"""
    return accuracy_score(y_test, y_pred) * 100


def matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)


def report(y_test, y_pred):
    return classification_report(y_test, y_pred)


def curve(y_test, y_pred, y_prob):
    auc_roc = roc_auc_score(y_test, y_pred)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    return (auc_roc, roc_auc, false_positive_rate, true_positive_rate)


def curve_plot(x, roc_auc, false_positive_rate, true_positive_rate):
    plt.figure(figsize=(10, 10))
    plt.title(x)
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="red",
        label="AUC = %0.2f" % roc_auc,
    )
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.axis("tight")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
