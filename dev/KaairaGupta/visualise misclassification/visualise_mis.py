import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import pandas as pd

def visualise_misclassification(classes,y_predict,y_actual):
    """
    This function helps to visualise misclassification in the data using confusion matrix
    input: Arrays of classes in which data is classified, y actual, and y predicted
    Output: A plt of fraction of misclassifed dats vs the true labels
    """
    confusion_mtx = confusion_matrix(y_actual, y_predict)
    label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    plt.bar(classes,label_frac_error)
    plt.xlabel('True Label')
    plt.ylabel('Fraction classified incorrectly')
    
