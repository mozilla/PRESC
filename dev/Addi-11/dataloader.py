""" importing required dependencies """

from IPython.display import HTML 
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

""" read data file """
address = "../../datasets/eeg.csv"
df = pd.DataFrame(pd.read_csv(address))
    
    
""" get some basic information about the dataset """
def data_info():
    print("\nFirst 5 rows of datasets: ")
    display(df.head(5))# first 5 rows of dataset
    print("\nLast 5 rows of datasets: ")
    display(df.tail(5)) # last 5 rows of dataset
    print("\nInformation of each column: \n")
    display(df.info()) # info of each column of dataset
    print("\nChecking for null values: \n")
    print(df.isnull().sum()) # sum of all null values in a dataset for preprocessing    

""" splitting the data """
def train_test_split_data(test_size):
    y = df.Class
    X = df.drop('Class',axis = 1)
    random_state = 40
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

    return X_train, X_test,y_train, y_test
