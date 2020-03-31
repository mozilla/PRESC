import pandas as pd
import numpy as np
from IPython.core.display import display


def read(filename):
    """
	This method takes the file path as input string 
	It returns the dataframe 
	"""
    df = pd.read_csv(filename)
    return df


def view_unique_values(dataframe):
    """
	This method takes a dataframe as input
	It prints the number of unique values in each column/feature
	"""
    for column in dataframe.columns:
        print(
            "Unique values in Column {} are {}".format(
                column, len(dataframe[column].unique())
            )
        )


def view_columns(dataframe):
    """
	This method takes a dataframe as input 
	It displays all the column names along with their data types 
	"""
    for column, datatype in dict(dataframe.dtypes).items():
        print("{} column is of type {}".format(column, datatype))


def view_label_distribution(target):
    """
	This method takes the target/lable Series as input 
	It displays the number of rows/samples belonging to that label
	"""
    for label, count in dict(target.value_counts()).items():
        print("{} samples have label {}".format(count, label))


def view_stats(dataframe):
    """
	This method takes a dataframe as input 
	It prints the important stats like max value/min value etc for each feature/column in the dataframe
	"""
    display(dataframe.describe())
