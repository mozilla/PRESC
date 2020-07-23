import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


def misclass_rate_feature(test_dataset, test_dataset_misclassifications, 
                          feature, bins=10):
    """Computes the misclassification rate as a function of the feature values.
    
    This function allows to compute the misclassification rate of a sample for
    the values of any particular feature. 
    
    The function allows any binning for this calculation, which means that
    regularly spaced binnings, disparately spaced binnings that correspond to
    sets of an equal amount of data points (such as quartiles, deciles, or 
    n-tiles), or any other arbitrary irregular binning can be used.
    
    When the full dataset with all points does not have any data point in an 
    interval corresponding to a certain bin, the function yields a "nan" value 
    for the misclassification rate to prevent a zero division error and also to 
    distinguish the bins without information from the bins with a zero 
    misclassification rate. The same happens with the standard deviation when
    either the full dataset with all points or the dataset with only the 
    misclassified points do not have any data point in a certain bin interval.  
    
    Parameters:
        test_dataset: Dataset with all the data points.
        test_dataset_misclassifications: Dataset with only the misclassified 
            points.
        feature: Column name in the dataset of the feature.
        bins (int, list, str): 
            * If an integer, it divides the feature scale in regularly spaced 
            bins (default value is 10).
            * If the string "quartile" is used, then it automatically computes  
            the appropriate bin edge postions to optimize for a quartile 
            grouping. 
            * If any other feature intervals are needed, then a list of the 
            feature values corresponding to the positions separating the bins 
            and including the outermost edges must be provided. 

    Returns:
        A list with three elements which correspond to:
            1) The edges of the bins in the feature scale.
            2) The misclassification rate in each bin.
            3) The standard deviation of the misclassification rate in that bin.
    """
    # Computes position of bin edges for quartiles or deciles
    if bins == "quartiles":
        bins = compute_tiles(test_dataset, feature, tiles=4)
    elif bins == "deciles":
        bins = compute_tiles(test_dataset, feature, tiles=10)
        
    # Histogram of all points
    total_histogram = np.histogram(test_dataset[feature], bins)

    # Histogram of misclassified points
    misclass_histogram = np.histogram(test_dataset_misclassifications[feature], 
                                      total_histogram[1], bins)
    
    # Compute misclassification rate
    
    # The standard deviation in a counting experiment is N^(1/2).
    # According to error propagation the error of a quotient X=M/N is:
    # ErrorX = X(ErrorM/M + ErrorN/N), 
    # here, Error_rate = rate*(M^(-1/2)+N^(-1/2))

    misclass_rate_histogram = np.copy(misclass_histogram)

    rate = []
    standard_deviation = []
    for index in range(len(total_histogram[0])):
        if total_histogram[0][index] != 0:
            index_rate = misclass_rate_histogram[0][index]/total_histogram[0][index]
            rate += [index_rate]
            if misclass_rate_histogram[0][index] != 0:
                standard_deviation += [index_rate*( 
                                    total_histogram[0][index]**(-1./2) +
                                    misclass_rate_histogram[0][index]**(-1./2))]
            else:
                standard_deviation += [float('nan')]
        else:
            rate += [float('nan')]
            standard_deviation += [float('nan')]
    misclass_rate_histogram[0] = rate

    return([misclass_rate_histogram[1], misclass_rate_histogram[0], standard_deviation])



def show_misclass_rate_feature(test_dataset, test_dataset_misclassifications, 
                               feature, bins=10):
    """Displays the misclassification rate for the values of a certain feature. 

    Parameters:
        test_dataset: Dataset with all the data points.
        test_dataset_misclassifications: Dataset with only the misclassified 
            points.
        feature: Column name in the dataset of the feature.
        bins (int, list): 
            * If an integer, it divides the feature scale in regularly spaced 
            bins (default value is 10).
            * If the string "quartiles" or "deciles" is used, then it 
            automatically computes the appropriate bin edge postions to optimize 
            for a quartile grouping. 
            * If any other feature intervals are needed, then a list of the 
            feature values corresponding to the positions separating the bins 
            and including the outermost edges must be provided. 
    """
    # Computes position of bin edges for quartiles or deciles
    if bins == "quartiles":
        bins = compute_tiles(test_dataset, feature, tiles=4)
    elif bins == "deciles":
        bins = compute_tiles(test_dataset, feature, tiles=10)

    misclass_rate_histogram = misclass_rate_feature(test_dataset, test_dataset_misclassifications, 
                                                    feature, bins=bins)
    width = [misclass_rate_histogram[0][index+1]-misclass_rate_histogram[0][index] 
                for index in range(len(misclass_rate_histogram[1]))]
    width_percentage = 1
    width_interval = [bin*width_percentage for bin in width]
    plt.ylim(0,1)
    plt.xlabel(feature)
    plt.ylabel("Misclassification rate")
    plt.bar(misclass_rate_histogram[0][:-1], misclass_rate_histogram[1], yerr=misclass_rate_histogram[2],
            width=width_interval, bottom=None, align='edge', edgecolor="white", linewidth=2)
    plt.show()



def show_misclass_rates_features(test_dataset, test_dataset_misclassifications, 
                                 bins=10):
    """Displays the misclassification rate for the values of each feature.

    Parameters:
        test_dataset: Dataset with all the data points.
        test_dataset_misclassifications: Dataset with only the misclassified 
            points.
        bins (int, list):
            * If an integer, it divides the feature scale in regularly spaced 
            bins (default value is 10).
            * If the string "quartile" is used, then it automatically computes  
            the appropriate bin edge postions to optimize for a quartile 
            grouping. 
            * If any other feature intervals are needed, then a list of the 
            feature values corresponding to the positions separating the bins 
            and including the outermost edges must be provided.    
    """
    # List of features
    feature_list = list(test_dataset.columns)[:-1]

    # Computes position of bin edges for quartiles or deciles
    for feature in feature_list:
        if bins == "quartiles":
            bins_feature = compute_tiles(test_dataset, feature, tiles=4)
        elif bins == "deciles":
            bins_feature = compute_tiles(test_dataset, feature, tiles=10)
        else:
            bins_feature = bins

        show_misclass_rate_feature(test_dataset, 
                                   test_dataset_misclassifications, 
                                   feature, bins=bins_feature)



def compute_tiles(dataset, feature, tiles = 4):
    """Computes optimal feature values to obtain quartiles, deciles, n-tiles...
    
    This function tries to determine the optimal feature value ranges in order  
    to obtain groups of data of similar sizes (i.e. with an equal amount of 
    samples), despite corresponding to feature intervals of different sizes.
    
    Very often this is not strictly possible. In particular, when the precision 
    of the feature is small and many data points share the same feature values
    (i.e. the feature behaves as pseudo-discrete). In this case, these large 
    subsets of data points sharing the same value either get all counted in one 
    bin or they get all counted in another. Which makes it impossible to 
    perfectly equilibrate the different groups.
    
    To arbitrarily split between two contiguous bins a subset of data points 
    with the same feature value is not acceptable if different histograms and 
    distributions have to be compared, or if normalization or other operations
    among them have to be carried out.
    
    Parameters:
        dataset (DataFrame): Data to try to chop in equal sets.
        feature: Column name in the dataset of the feature.
        tiles (int): Number of equally-sized groups into which to try to divide 
            the sample. For quartiles use 4, for deciles use 10, etc. Default 
            value is 4.
    
    Returns:
        edge_values (list): List of the optimal edge positions.
    """
    points_per_tile = int(len(dataset)/float(tiles))
    ordered_dataset = dataset.sort_values(by=feature)
    
    edge_values = [ordered_dataset[feature].min()]
    
    for index in range(tiles-1):
        edge_values += [ordered_dataset[feature][index*points_per_tile:(index+1)*points_per_tile-1].max()]
        
    edge_values += [ordered_dataset[feature].max()]
    
    return edge_values


def show_tiles_feature(dataset, feature, tiles = 4):
    """Plots the best attempt to obtain quartiles/deciles/n-tiles for a feature.
    
    This function shows the different tiles computed for one of the features in 
    order to assess whether the data that is being used really allows for that 
    particular number of tiles to have a similar size or not.
    
    Parameters:
        dataset (DataFrame): Data to try to chop in equal sets.
        feature: Column name in the dataset of the feature.
        tiles (int): Number of equally-sized groups into which to try to divide 
            the sample. For quartiles use 4, for deciles use 10, etc. Default 
            value is 4. 
    """
    tiles_feature = compute_tiles(dataset, feature, tiles=tiles)
    total_histogram = np.histogram(dataset[feature], bins=tiles_feature)
    
    width = [total_histogram[1][index+1]-total_histogram[1][index]
         for index in range(len(total_histogram[0]))]
    width_percentage = 1
    width_interval = [bin*width_percentage for bin in width]
    
    plt.xlabel(feature)
    plt.ylabel("counts")
    plt.bar(total_histogram[1][:-1], total_histogram[0], width=width_interval, 
            bottom=None, align='edge', edgecolor="white", linewidth=3)
    plt.show()


def show_tiles_features(test_dataset, tiles=4):
    """Plots the best attempt to obtain quartiles/deciles/n-tiles for all features.
    
    This function shows the different tiles computed for each one of the 
    features in order to assess whether the data that is being used really 
    allows for that particular number of tiles to have a similar size for that 
    feature or not.
    
    Parameters:
        dataset (DataFrame): Data to try to chop in equal sets.
        tiles (int): Number of equally-sized groups into which to try to divide 
            the sample. For quartiles use 4, for deciles use 10, etc. Default 
            value is 4. 
    """
    # List of features
    feature_list = list(test_dataset.columns)[:-1]
    
    for feature in feature_list:
        if tiles == "quartiles":
            tiles_feature = 4
        elif tiles == "deciles":
            tiles_feature = 10
        else:
            tiles_feature = tiles

        show_tiles_feature(test_dataset, feature, tiles=tiles_feature)



def main():
    pass

if __name__ == "__main__":
    main()
