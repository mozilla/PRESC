import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def separate_target(dataframe):
    """
	This method takes the original dataframe read from the csv file as input 

	It separates the class/label/target from the rest of the data values/features/columns and returns them spearately as a series & dataframe respectively
	"""
    target = dataframe.iloc[:, -1]
    dataframe = dataframe.drop(dataframe.columns[-1], axis=1)
    return dataframe, target


def min_max_normalization(dataframe):
    """
	This module takes the dataframe outputted by seperate_target method and performs min-max normalization on the data values to 
	get them between 0 to 1. Thus avoiding any overfitting by the models on features with larger values
	
	It returns a dataframe with normalized values 
	"""
    for column in dataframe.columns:
        dataframe[column] = (dataframe[column] - dataframe[column].min()) / (
            dataframe[column].max() - dataframe[column].min()
        )
    return dataframe


def standardization(dataframe):
    """
	This module takes the dataframe outputted by seperate_target method and performs standardization to change
	the mean to 0 and variance to 1 for all the features. Thus avoiding any overfitting by the models on features with larger scattered/shifted values
	
	It returns a dataframe with standardized values 
	"""
    scaler = StandardScaler()
    result = scaler.fit_transform(dataframe.iloc[:, dataframe.columns != "Class"])
    standardized_df = pd.DataFrame(
        result, columns=[col for col in dataframe.columns if col != "Class"]
    )
    return standardized_df


def dim_reduction_PCA(dataframe, components=6):
    """
	This method performs Principal Component Analysis to perform dimensionality reduction on the feature set that are correlated 
	It takes the dataframe outputted either by standardization or min-max-normalization as input, however standardization dataframe is recommended for
	get accurate results 
	It takes the number of principal components as input from the user with the default set to 6

	It returns a dataframe of the principal components
	"""
    column_names = [
        "PC" + str(i) for i in range(1, components + 1)
    ]  # geenrates names for PC columns
    pcaa = PCA(n_components=components)
    PCC = pcaa.fit_transform(dataframe)
    PC_df = pd.DataFrame(data=PCC, columns=column_names)
    print(
        "Amount of information preserved in {} principle components is {:.2f}%".format(
            components, np.sum(pcaa.explained_variance_ratio_) * 100
        )
    )
    return PC_df


def feature_correlation_heatmap(dataframe):
    """
	This method takes .It helps visualize the correlation between the different features 

	It outputs a heatmap that gives a detailed visualization of the correlation between all the pairs of features
	"""
    stored = dataframe.corr()
    f, ax = plt.subplots(figsize=(11, 9))
    ax = sns.heatmap(stored, annot=True, cmap="YlGnBu")
