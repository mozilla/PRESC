#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[4]:


clf = RandomForestClassifier()


# In[5]:


#function to split the dataset into train and test sets
def split(df, target, test):
    #df = dataframe
    #target = target column
    # test = proportion of test set (float number)
    #stratified shuffle split because the data is not balanced
    split = StratifiedShuffleSplit(n_splits = 1, test_size=test, random_state=42)
    for train_index, test_index in split.split(df, df[target]):
        train = df.loc[train_index]
        test = df.loc[test_index]
        
        x = train.drop([target], axis=1)
        #train target
        y = train[target]
    
        #test feature
        a = test.drop([target], axis=1)
        #test target
        b = test[target]
        
        return train, test


# In[6]:


#function to return the accuracy of a model
def accuracy(df, target, test):
    train,test = split(df, target, test)
    
    x = train.drop([target], axis=1)
    #train target
    y = train[target]
    
    #test feature
    a = test.drop([target], axis=1)
    #test target
    b = test[target]
    
    clf.fit(x,y)
    prediction = clf.predict(a)
    
    accuracy = accuracy_score(prediction, b)
    return accuracy * 100


# In[7]:


#returns the confusion matrix 
def conf(df, target, test):
    train,test = split(df, target, test)
    
    x = train.drop([target], axis=1)
    #train target
    y = train[target]
    
    #test feature
    a = test.drop([target], axis=1)
    #test target
    b = test[target]
    
    clf.fit(x,y)
    prediction = clf.predict(a)
    c_m = confusion_matrix(b, prediction)
    return c_m


# In[8]:


def precision_(df, target, test):
    
    train,test = split(df, target, test)
    
    x = train.drop([target], axis=1)
    #train target
    y = train[target]
    
    #test feature
    a = test.drop([target], axis=1)
    #test target
    b = test[target]
    
    clf.fit(x,y)
    prediction = clf.predict(a)
    
    p = precision_score(b, prediction, average='micro')
    return p


# In[9]:


def recall_(df, target, test):
    
    train,test = split(df, target, test)
    
    x = train.drop([target], axis=1)
    #train target
    y = train[target]
    
    #test feature
    a = test.drop([target], axis=1)
    #test target
    b = test[target]
    
    clf.fit(x,y)
    prediction = clf.predict(a)
    
    r = recall_score(b, prediction, average='micro')
    return r


# In[10]:


def f1_(df, target, test):
    
    train,test = split(df, target, test)
    
    x = train.drop([target], axis=1)
    #train target
    y = train[target]
    
    #test feature
    a = test.drop([target], axis=1)
    #test target
    b = test[target]
    
    clf.fit(x,y)
    prediction = clf.predict(a)
    
    f = f1_score(b, prediction, average='micro')
    return f

