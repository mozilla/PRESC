
# importing required dependencies
from IPython.display import HTML
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

# read data file
address = "../../datasets/eeg.csv"
df = pd.DataFrame(pd.read_csv(address))

def data_info():
    """
	Displays basic information about a dataset :
		first 5 rows of dataset
		last 5 rows of dataset
		info of each column of dataset
		description on the dataset
		sum of all null values in a dataset for preprocessing
	"""
    print("\nFirst 5 rows of datasets: ")
    display(df.head(5))
    print("\nLast 5 rows of datasets: ")
    display(df.tail(5))
    print("\nDescription of each column: \n")
    display(df.describe())
    print("\nInformation of each column: \n")
    display(df.info())
    print("\nChecking for null values: \n")
    print(df.isnull().sum())

def data_visuals():
    """
	Visualises basic infromation about the dataset.
	This function is specific to the eeg dataset.
	"""
    df.Class.unique()
    sns.countplot(df.Class)

def get_x_y():
    y = df.Class
    x = df.drop("Class", axis=1)
    return x, y

def train_val_test_split_data(test_size):
    """
	Splits the data into 3 portions, training, validation and testing.
	
	Parameters:
		test_size : float-range(0,1), the ratio of test size to total size.

	Returns:
		x_train : array-like, shape(n_train_samples, n_features)
		x_val : of length n_validation_samples
		x_test : array-like, shape(n_test_samples, n_features)
		y_train : of length n_train_samples
		y_val : of length n_validation_samples
		y_test : of length n_test_samples
	"""

    x, y = get_x_y()
    random_state = 40

    # dividing training set into training and testing segments
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # further dividing training set into training and validation segments
    val_size = 0.2  # set the validation size 0.2 of the training set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
