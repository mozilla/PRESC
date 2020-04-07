"""
Importing dataset and pre-processing it
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from math import sqrt, floor

# loading the data
def load_dataset():
    """
    Return dataframe using pandas.
    """
    d = pd.read_csv("../../datasets/winequality.csv")
    return d

def visualize_data(d):
    """
    Visualize the dataset in terms of:
        -distribution of target labels (recommend: True/ False).\n
        -observing available features in the dataset.\n
        -pairplots of all the features with one-another.\n
        -correlational matrix.\n

    Observe patterns in the data with correlation matrix.

    """
    d["recommend"].value_counts().plot(kind="bar")

    print("List of features in the dataset:")
    features = list(d.columns.values)[:-1]
    print(features)
    print("\n\n")

    sns.set(style="ticks", color_codes=True)
    sns.pairplot(d, vars=features, hue="recommend", diag_kind='hist')

    # Correlation Matrix
    corr = d.corr()

    # Visualising the correlation matrix
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    ticks = np.arange(len(features))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    names = list(d)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.title("Correlation Matrix Visualization", size=16, y=1)
    plt.xticks(rotation=45)
    sns.heatmap(corr, annot=True, ax=ax)

def FeaturesLabels(d):
    """
    Break the dataset into features and target labels.
    """
    x = d.iloc[:, :-2]  # features ('quality' has been excluded being
    # a direct indicator of target class)
    y = d.iloc[:, -1]  # target labels (recommend: True or False)
    return x, y

def PreProcess_and_Split(d, ts = 0.4):
    """
    This function pre-processes the data and splits into 
    training and test sets.
    
    Split the data using stratified splitting to maintain similar 
    proportion of target labels in both training and testing sets.
    
    Split Ratio : Test size of 0.4 has been used maintaining a 
    60-40 ratio for training and test data, after a series of hit and trials 
    with different test sizes. A peak in accuracy was observed with all models 
    around the test size of 0.4 for used dataset.
    
    Changing these parameters may lead to underfitting or overfitting of model.
    
    Using StandardScalar, all the data has been brought to the same scale to 
    reduce complexity in calculation.
    It transforms the data in such a manner that it has mean as 0 and standard
    deviation as 1.
    """
    x, y = FeaturesLabels(d)

    # Splitting the dataset

    sss = StratifiedShuffleSplit(n_splits=1, test_size=ts, random_state=0)
    sss.get_n_splits(x, y)
    print(sss)
    for train_index, test_index in sss.split(x, y):
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Feature scaling

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test, y_train, y_test

def undersample(d, size = 0.5):
    '''
    This function undersamples the data in such a way that the final data sample has equi-distribution of target labels throughout. 
    '''
    
    majority_class_indices = d[d['recommend']==0].index
    minortity_class_indices = d[d['recommend']==1].index
    
    #required sample length with proportion of minority class data points = size (default = 0.5)
    req_len = floor((1 - size)/size * len(d[d['recommend']]==1))
    
    # List of random indices of majority class
    random_majority_indices = np.random.choice(majority_class_indices, req_len, replace=True)
    
    # creating the Undersample
    under_sample_indices = np.concatenate([minortity_class_indices, random_majority_indices])
    under_sample = d.loc[under_sample_indices]
    
    # Renaming indices serially
    ind = list(range(0, len(under_sample)))
    under_sample = under_sample.set_index([pd.Index(ind)])
    return under_sample