#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
<<<<<<< HEAD


# In[2]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score


# In[3]:
=======
import seaborn as sns


# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier


# In[ ]:
>>>>>>> parent of 89dd16e... moved to misclassification visualization folder


clf = RandomForestClassifier()


<<<<<<< HEAD
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
=======
# This function takes in the test set, the target and the prediction and returns the total length of the test set and how many datapoints were missclassified

# In[ ]:


def misclassified_data(df, target, prediction):
    misclassified = df[df[target] != df[prediction]]
    print('Total Test Set: {}'.format(len(df)))
    print("Number of Misclassified Datapoints: {}".format(len(misclassified)))


# In[ ]:


def plot_misclassified(df, target):
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
    
        #fit the random forest model
        clf.fit(x,y)
    
        #make prediction and add to the test dataframe
        predictions = clf.predict(a)
        test['predict'] = predictions
        
        #get the prediction probability and make it a dataframe
        probability = clf.predict_proba(a)
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
>>>>>>> parent of 89dd16e... moved to misclassification visualization folder

