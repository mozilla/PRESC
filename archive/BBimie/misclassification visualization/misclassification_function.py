#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score


# In[3]:


clf = RandomForestClassifier()


# This function splits the dataset into train and test sets and then asigns features and target of each set to x & y respectively
# 
# > Inputs
# 
#  - df = dataframe
#  
#  - target = prediction target
#  
#  - proportion of test set (20% will be written as 0.2)

# In[4]:


def split(df, target, test):
    split = StratifiedShuffleSplit(n_splits = 1, test_size=test, random_state=42)
    for train_index, test_index in split.split(df, df[target]):
        train = df.loc[train_index]
        test = df.loc[test_index]
        
        x_train = train.drop([target], axis=1)
        y_train = train[target]
    
        x_test = test.drop([target], axis=1)
        y_test = test[target]
        
        return x_train, y_train, x_test, y_test


# This function fits the model with the train features and target column, and performs prediction on the test feature
# 
# - x_train, y_train,  x_test are the outputs from the split function

# In[5]:


def predict(x_train, y_train,  x_test):

    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    
    return prediction


# - this function returns the number of misclassified datapoints 

# In[23]:


def missclassified(df, prediction, y_test):
    
    misclassified = y_test != prediction
    
    return print('This is the number of misclassified data: ', len(misclassified), '\n' 
            '{} % of the data was misclassified'.format((len(misclassified)/len(df))*100))


# - this function creates a confusion matrix dataframe

# In[8]:


def confusion_mat(prediction, y_test):
    c_m = confusion_matrix(prediction, y_test)
    labels = [3,4,5,6,7,8,9]
    
    c_m = pd.DataFrame(c_m, index=labels, columns=labels)
    
    return c_m


# - this function plots a barchart showing the number of misclassified classes for each wine quality class

# In[31]:


def plot_misclassified(c_m):
    for label in c_m.columns:
        c_m.at[label, label] = 0
        
    ax = c_m.plot(kind="bar", width=1.5)
    ax.locator_params(axis="y", integer=True)
    ax.set_xlabel("Quality classes")
    ax.set_ylabel("Number of Misclassified Data")
    ax.set_title('Misclassified Data')
    plt.show()

