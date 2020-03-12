# importing necessary libraries
import os
import pandas as pd
import numpy as np
import pandas_profiling
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,\
classification_report,roc_curve,confusion_matrix

class DefineModules():
    '''
    Defines the different functions that will be called during
    execution
    '''

    # reading the data
    def read_data(folder, file_name):
        '''
        This function reads the csv file from the folder it is stored in 
        and returns a dataframe.

        Parameters: main directory, file name

        Returns: dataframe    
        '''
        data = pd.read_csv('../../{}/{}.csv'.format(folder, file_name))
        
        # setting index for the data
        data.set_index('id', inplace =True)

        return data

    # studying the data
    def study_data(data):
        '''
        This function helps know the various details of the data like, 
        number of null values, skewness, distribution, correlations, 
        missing values, interactions, data type, unique values etc.

        Parameters: dataframe

        Returns: report object


        '''
        profile = pandas_profiling.ProfileReport(data)

        return profile
