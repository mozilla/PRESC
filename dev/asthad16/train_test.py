#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from data_preprocess import data_split
from evaluation import *
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd


# In[ ]:


def split_table(classifier, x, y1):
    """computes an evaluation metric over a grid of train/test split proportions from 0 to 100%"""
    train_set = []
    test_set = []
    accuracy = []
    f1 = []
    precision = []
    recall = []
    # doing data split
    for i in range(5, 101, 1):
        j = 100 - i
        if i <= j:
            split = i / 100
            X_train, X_test, y_train, y_test = data_split(x, y1, split)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            y_prob = classifier.predict_proba(X_test)[:, 1]
            # creating list having train-test ratio
            train_set.append(j)
            test_set.append(i)
            accuracy.append(acc_score(y_test, y_pred))
            f1.append(f1_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred))
            recall.append(recall_score(y_test, y_pred))

    split_table = pd.DataFrame(
        np.column_stack([train_set, test_set, accuracy, f1, precision, recall]),
        columns=[
            "train_set(%)",
            "test_set(%)",
            "accuracy",
            "f1_score",
            "precision_score",
            "recall_score",
        ],
    )
    return split_table
