import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as skl
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from pandas import set_option
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def showSimplePlot(data):
    '''
    Plots a simple scatterplot of the x and y values.
    Args: The data in the form of dictionary right now. A list of y values for each x
    '''
    yavg = list()
    xax = list()
    yax = list()
    for key, value in data.items():
        yavg.append(np.mean(value))
        xax.append(key)
        yax.append(value)
    plt.plot(xax, yax, 'o', color='black') # y values in black
    plt.plot(xax, yavg, 'o', color='red') # avg y values in red
    
def violinAndBoxPlots(data):
    '''
    Plots violin and Box plots for the data
    Args: The data in the form of dictionary right now. A list of y values for each x
    '''
    yavg = list()
    xax = list()
    yax = list()
    for key, value in data.items():
        yavg.append(np.mean(value))
        xax.append(key)
        yax.append(value)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    all_data = yax

    # plot violin plot
    axs[0].violinplot(all_data,
                      showmeans=True,
                      showmedians=False)
    axs[0].set_title('Violin plot')

    # plot box plot
    axs[1].boxplot(all_data, usermedians=yavg)
    axs[1].set_title('Box plot')

    # adding horizontal grid lines
    for ax in axs:
        ax.yaxis.grid(True)
        ax.set_xticks(xax)
        ax.set_xlabel('Four separate samples')
        ax.set_ylabel('Observed values')

    # add x-tick labels
    plt.setp(axs, xticks=xax,
             xticklabels=['x1', 'x2', 'x3', 'x4'])
    plt.show()