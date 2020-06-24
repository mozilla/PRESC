import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


'''Function to read dataset from file'''
def read_data(path=None):
    if path is None:
        print('Path not specified.')
        return None, None, None, None, None
    elif os.path.exists(path) is False:
        print('File not found.')
        return None, None, None, None, None
    dataset=pd.read_csv(path)
    return dataset

    
'''Function to split the dataset as train and test dataset'''
def split_data(x=None, y=None, train_size=None):
    train_size=0.25
    if x is None or y is None:
        return None, None, None, None
    elif train_size is None:
        x_train, x_test, y_train, y_test = train_test_split(x, y)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

    return x_train, x_test, y_train, y_test
