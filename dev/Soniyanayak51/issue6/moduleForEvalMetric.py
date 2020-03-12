import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def showSimplePlot(xax, yax):
    '''
    Plots a simple scatterplot of the x and y values.
    Args: The data in the form of dictionary right now. A list of y values for each x
    '''
    yavg = list()
    for y in yax:
        yavg.append(np.mean(y))
    plt.plot(xax, yax, 'o', color='black')  # y values in black
    plt.plot(xax, yavg, 'o', color='red')  # avg y values in red


def violinAndBoxPlots(xax, yax):
    '''
    Plots violin and Box plots for the data
    Args: The data in the form of dictionary right now. A list of y values for each x
    '''
    yavg = list()
    for y in yax:
        yavg.append(np.mean(y))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))

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
        ax.set_xlabel('C values')
        ax.set_ylabel('Accuracy')

    # add x-tick labels
    plt.setp(axs, xticks=xax,
             xticklabels=xax)
    plt.show()


def generateDatasetUsingC(features, y):
    '''
    generates a dataset using repeated k-fold cv

    Args: Features and labels
    Returns: all_scores - List of lists of scores for each c, accuracy_parameter_sweep - mean array, std_parameter_sweep - std array
    '''
    accuracy_parameter_sweep = []
    std_parameter_sweep = []
    c_sweep = np.power(2, np.linspace(-4, 4, 5))  # list of values of hyper-parameter C
    all_scores = []  # keep scores here
    scores_c = []
    for c in c_sweep:
        clf = SVC(gamma='auto', verbose=True, C=c)
        scores_c = cross_val_score(clf, features, y, cv=10)
        accuracy_parameter_sweep.append(np.mean(100 * scores_c))
        std_parameter_sweep.append(np.std(100 * scores_c))
        all_scores.append(scores_c)
        print(scores_c)
    return all_scores, accuracy_parameter_sweep, std_parameter_sweep


def meanVarianceSpreadPlot(c_sweep, accuracy_parameter_sweep, std_parameter_sweep):
    '''
    Plots the variation of mean of accuracy for each C along with the spread of variance.
    Args: c_sweep - list of values of C, accuracy_parameter_sweep - mean of scores for each C*100, std_parameter_sweep-variance of accuracies for each C
    accuracy_parameter_sweep and std_parameter_sweep returned by function generateDatasetUsingC
    '''
    plt.fill_between(c_sweep, np.array(accuracy_parameter_sweep) - np.array(std_parameter_sweep),
                     np.array(accuracy_parameter_sweep) + np.array(std_parameter_sweep), facecolor='xkcd:light pink',
                     alpha=0.7)
    plt.semilogx(c_sweep, accuracy_parameter_sweep, color='xkcd:red', linewidth=4)
    plt.xlabel('C')
    plt.ylabel('Accuracy (%)')
    plt.title('SVM Accuracy vs. Hyper-parameter C')
    plt.grid(True, which='both')
    plt.tight_layout()