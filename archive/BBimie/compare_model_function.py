#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


# In[7]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score


# In[8]:


clf = RandomForestClassifier()
knn = KNeighborsClassifier(n_neighbors = 5)


# In[16]:


def plot_probability(df, target, model):
    #stratified shuffle split because the data is not balanced
    split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df[target]):
        train = df.loc[train_index]
        test = df.loc[test_index]
        
        #train feature
        x = train.drop([target], axis=1)
        #train target
        y = train[target]
    
        #test feature
        a = test.drop([target], axis=1)
        #test target
        b = test[target]
    
        #fit the model
        model.fit(x,y)
    
        #make prediction and add to the test dataframe
        predictions = model.predict(a)
        test['predict'] = predictions
        
        #get the prediction probability and make it a dataframe
        probability = model.predict_proba(a)
        probability_df = pd.DataFrame(probability)
        
        #get the maximum probability for each prediction
        max_prob = probability_df.max(axis=1)
        
        #change the indexing of the test set to start from zero
        test.index = np.arange(0, len(test))
        
        #concatenate test set and max_prob
        full = pd.concat([test, max_prob], axis=1)
        full = full.rename(columns={0:'probability'})
        
        #visualize the maximum probability of prediction 
        sns.catplot(x='predict', y="probability", hue="quality", data=full, height=6, kind="bar")


# In[ ]:


def plot_probability_misclassification(df, target, model):
    #stratified shuffle split because the data is not balanced
    split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df[target]):
        train = df.loc[train_index]
        test = df.loc[test_index]
        
        #train feature
        x = train.drop([target], axis=1)
        #train target
        y = train[target]
    
        #test feature
        a = test.drop([target], axis=1)
        #test target
        b = test[target]
    
        #fit the model
        model.fit(x,y)
    
        #make prediction and add to the test dataframe
        predictions = model.predict(a)
        test['predict'] = predictions
        
        #get the prediction probability and make it a dataframe
        probability = model.predict_proba(a)
        probability_df = pd.DataFrame(probability)
        
        #get the maximum probability for each prediction
        max_prob = probability_df.max(axis=1)
        
        #change the indexing of the test set to start from zero
        test.index = np.arange(0, len(test))
        
        #concatenate test set and max_prob
        full = pd.concat([test, max_prob], axis=1)
        full = full.rename(columns={0:'probability'})
        
        misclassified = full[full['predict'] != full['quality']]
        
        #visualize the maximum probability of prediction 
        sns.catplot(x='predict', y="probability", hue="quality", data=misclassified, height=6, kind="bar")

