"""
Importing dataset and pre-processing it
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns

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
        -pairplotss of all the features with one-another.\n
        -correlational matrix.\n

    Observe patterns in the data with correlation matrix.

    """
    d["recommend"].value_counts().plot(kind="bar")

    print("List of features in the dataset:")
    features = list(d.columns.values)[:-1]
    print(features)
    print("\n\n")

    sns.set(style="ticks", color_codes=True)
    sns.pairplot(
        d, vars=features, hue="recommend",
    )

    # Correelation Matrix
    corr = d.corr()
    print("Correlation Matrix:")
    print(corr)
    print("\n\n")

    # Visualising the correlation matrix
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot()
    cax = ax.matshow(corr, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(features), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    names = list(d)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.title("Correlation Matrix Visualization", size=16, y=1.2)
    plt.xticks(rotation=45)
    plt.show()

def FeaturesLabels(d):
    """
    Break the dataset into features and target labels.
    """
    x = d.iloc[:, :-2]  # features ('quality' has been excluded being
    # a direct indicator of target class)
    y = d.iloc[:, -1]  # target labels (recommend: True or False)
    return x, y

def PreProcess_and_Split(d):
    """
    This function pre-processes the data and splits into 
    training and test sets.
    
    Split the data using stratified splitting to maintain similar 
    proportion of target labels in both training and testing sets.
    
    Split Ratio : Test size of 0.2 has been used maintaining a 
    reasonable 80-20 ratio for training and test data. 
    
    Changing these parameters may lead to underfitting or overfitting of model.
    
    Using StandardScalar, all the data has been brought to the same scale to 
    reduce complexity in calculation.
    It transforms the data in such a manner that it has mean as 0 and standard
    deviation as 1.
    """
    x, y = FeaturesLabels(d)

    # Splitting the dataset

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
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
