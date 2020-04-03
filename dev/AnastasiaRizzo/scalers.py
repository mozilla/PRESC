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
from scipy.stats import norm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, Normalizer


from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")



def scalers (X_train, y_train, X_test, y_test):
    
    '''
    This funcion applies different scalers on the data, 
    calculates the best scaler and plots the table with 
    the results
    
    '''
    
    KNN = KNeighborsClassifier()
    KNN_params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
    KNN1 = GridSearchCV(KNN, param_grid = KNN_params)
    KNN1.fit(X_train, y_train)
    KNN1.best_params_
    KNN1_pred = KNN1.predict(X_test)

    models = []
    models.append(("K-Nearest Neighbour:", KNeighborsClassifier(n_neighbors = 1)))
    names = []

    for name, model in models:

            # scalers
            scaler1 = StandardScaler()
            scaler2 = MinMaxScaler()
            scaler3 = MaxAbsScaler()
            scaler4 = RobustScaler()
            scaler5 = QuantileTransformer(output_distribution = 'normal')
            scaler6 = QuantileTransformer(output_distribution = 'uniform')
            scaler7 = PowerTransformer()
            scaler8 = Normalizer()

            # build and train models
            X_train_scaled1 = scaler1.fit_transform(X_train)
            X_test_scaled1 = scaler1.transform(X_test)
            model.fit(X_train_scaled1, y_train)
            pred1 = model.predict(X_test_scaled1)

            X_train_scaled2 = scaler2.fit_transform(X_train)
            X_test_scaled2 = scaler2.transform(X_test)
            model.fit(X_train_scaled2, y_train)
            pred2 = model.predict(X_test_scaled2)

            X_train_scaled3 = scaler3.fit_transform(X_train)
            X_test_scaled3 = scaler3.transform(X_test)
            model.fit(X_train_scaled3, y_train)
            pred3 = model.predict(X_test_scaled3)

            X_train_scaled4 = scaler4.fit_transform(X_train)
            X_test_scaled4 = scaler4.transform(X_test)
            model.fit(X_train_scaled4, y_train)
            pred4 = model.predict(X_test_scaled4)

            X_train_scaled5 = scaler5.fit_transform(X_train)
            X_test_scaled5 = scaler5.transform(X_test)
            model.fit(X_train_scaled5, y_train)
            pred5 = model.predict(X_test_scaled5)

            X_train_scaled6 = scaler6.fit_transform(X_train)
            X_test_scaled6 = scaler6.transform(X_test)
            model.fit(X_train_scaled6, y_train)
            pred6 = model.predict(X_test_scaled6)

            X_train_scaled7 = scaler7.fit_transform(X_train)
            X_test_scaled7 = scaler7.transform(X_test)
            model.fit(X_train_scaled7, y_train)
            pred7 = model.predict(X_test_scaled7)

            X_train_scaled8 = scaler8.fit_transform(X_train)
            X_test_scaled8 = scaler8.transform(X_test)
            model.fit(X_train_scaled8, y_train)
            pred8 = model.predict(X_test_scaled8)

            # accuracy calculation
            accuracy1 = accuracy_score(pred1, y_test)
            accuracy2 = accuracy_score(pred2, y_test)
            accuracy3 = accuracy_score(pred3, y_test)
            accuracy4 = accuracy_score(pred4, y_test)
            accuracy5 = accuracy_score(pred5, y_test)
            accuracy6 = accuracy_score(pred6, y_test)
            accuracy7 = accuracy_score(pred7, y_test)
            accuracy8 = accuracy_score(pred8, y_test)

            names.append(name)

    for i in range(len(names)):

            # visualisation table
            df = pd.DataFrame({"Model": [names[i], names[i], names[i], names[i], names[i], names[i], names[i], names[i]],
                               "Scaler": [scaler1, scaler2, scaler3, scaler4, scaler5, scaler6, scaler7, scaler8],
                               "Accuracy": [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8]})

            # print table
            display(df)

