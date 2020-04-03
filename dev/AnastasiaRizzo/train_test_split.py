#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn.metrics as metrics
from sklearn.metrics import (accuracy_score, 
                             f1_score,
                             precision_score, 
                             average_precision_score, 
                             recall_score
                            )
from sklearn.model_selection import (train_test_split, 
                                     GridSearchCV
                                    )

from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")


split_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
columns1 = ['X_train','X_test','Accuracy', 'Accuracy with hyper parameters', 'F1_score', 'F1_score with hyper parameters']
table = pd.DataFrame(columns = columns1)

random1 = [10, 20, 30, 42, 50, 60, 70, 80, 90]
columns2 = ['','Random State', 'Accuracy', 'Accuracy with hyper parameters', 'F1_score', 'F1_score with hyper parameters']
table1 = pd.DataFrame(columns = columns2) 



def split_train_test (X, y, split_size, columns1, table, index):
    
    '''
    This function calculates 'test_size' split 
    % for Train/Test data and plots the table 
    with the results
    
    '''
    
    for index in range(len(split_size)):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = split_size[index], random_state=0)

                KNN = KNeighborsClassifier()
                KNN.fit(X_train, y_train)
                KNN_pred = KNN.predict(X_test)
                KNN_accuracy = accuracy_score(KNN_pred, y_test)

                KNN_params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
                KNN1 = GridSearchCV(KNN, param_grid = KNN_params)
                KNN1.fit(X_train, y_train)
                KNN1_pred = KNN1.predict(X_test)
                KNN1_accuracy = accuracy_score(KNN1_pred, y_test)

                y_score = KNN.predict(X_test)
                f_score = f1_score(y_test, y_score)

                y_score_1 = KNN1.predict(X_test)         
                f_score_1 = f1_score(y_test, y_score_1)

                train_split = round((1-split_size[index])*100)
                test_split = round(split_size[index]*100)
                table.loc[index+1] = [train_split, test_split, KNN_accuracy,  KNN1_accuracy,f_score, f_score_1]

    display(table)  

    

def random_state (X, y, random1, columns2, table1, index):
 
    '''
    This function calculates 'random_state' 
    number for Train/Test data and plots the 
    table with the results
    
    '''
    
    for index in range(len(random1)):
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random1[index],test_size = 0.3)

                KNN = KNeighborsClassifier()
                KNN.fit(X_train, y_train)
                KNN_pred = KNN.predict(X_test)
                KNN_accuracy = accuracy_score(KNN_pred, y_test)

                KNN_params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
                KNN1 = GridSearchCV(KNN, param_grid = KNN_params)
                KNN1.fit(X_train, y_train)
                KNN1_pred = KNN1.predict(X_test)
                KNN1_accuracy = accuracy_score(KNN1_pred, y_test)

                y_score = KNN.predict(X_test)
                f_score = f1_score(y_test, y_score)

                y_score_1 = KNN1.predict(X_test)         
                f_score_1 = f1_score(y_test, y_score_1)
        
                random_split_train = 0
                random_split_test = round(random1[index])
                table1.loc[index+1] = [random_split_train, random_split_test, KNN_accuracy,  KNN1_accuracy, f_score, f_score_1]

    display(table1)  

