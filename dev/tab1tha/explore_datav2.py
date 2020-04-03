"""This is just an updated copy of explore_data. It is kept separate to prevent merge conflicts. """

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def raw(df):
    """This function provides insights into the
     structure of content in the dataset. It shows that target values are all clearly defined 
     and discrete. Hence, a classification model is appropriate. """
    print("*" * 50 + "HEAD" + "*" * 50)
    print(df.head())
    print("*" * 50 + "TAIL" + "*" * 50)
    print(df.tail())
    print("*" * 50 + "DESCRIBE" + "*" * 50)
    print(df.describe())
    print("*" * 50 + "INFO" + "*" * 50)
    print(df.info())
    print("*" * 100)


def raw_more(df):
    """This function goes a step further from the raw() function in easing the understanding of the 
    dataset and its data. It puts more emphasis on the interrelation between variables and gives more
    insight into the target variable"""
    print(df.Class.value_counts())
    print(df.groupby("Class").mean())
    sns.heatmap(df.corr())
    plt.show()


def graph_visualize(df, targetname):
    """This function plots a pair plot that provides a detailed view of interrelationships in
     the dataset """
    # visualisation
    plt.figure()
    sns.pairplot(df, hue=targetname)
    plt.show()


def violin_visualize(df):
    """This function investigates the distribution of observation results 
    among the target values by drawing a violin plot"""
    # create a figure instance
    fig = plt.figure()
    """create an axes instance and specifies appropriate coordinates.
     Values are just defaults chosen through tinkering. """
    ax = fig.add_axes([0, 0, 1, 1])
    # create the boxplot
    bp = ax.violinplot(df["label"])
    plt.show(bp)


def confu_matrix_display(conf_matrix, target, title):
    """This functions receives a confusion matrix and the list of unique target variable values as 
    parameters. The target values serve as a source of labels to ease understanding of the confusion 
    matrix that is plotted out as output. """
    target_vals = np.unique(target)
    df_conf_matrix = pd.DataFrame(conf_matrix, index=target_vals, columns=target_vals)
    sns.heatmap(df_conf_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.title(title + " confusion matrix")
    plt.show()
