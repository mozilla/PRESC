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
                          feature="", bins=10):

    # Histogram of all points (regular intervals)
    total_histogram = np.histogram(test_dataset[feature], bins)

    # Histogram of misclassified points (regular intervals)
    misclass_histogram = np.histogram(test_dataset_misclassifications[feature], 
                                      total_histogram[1], bins)
    
    # Compute misclassification rate
    misclass_rate_histogram = np.copy(misclass_histogram)
    
    # The standard deviation in a counting experiment is N^(1/2).
    # According to error propagation the error of a quotient X=M/N is:
    # ErrorX = X(ErrorM/M + ErrorN/N), 
    # here, Error_rate = rate*(M^(-1/2)+N^(-1/2))

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
                standard_deviation += [0.0]
        else:
            rate += [float('nan')]
            standard_deviation += [float('nan')]
    misclass_rate_histogram[0] = rate

    return([misclass_rate_histogram[1], misclass_rate_histogram[0], standard_deviation])



def show_misclass_rate_feature(test_dataset, test_dataset_misclassifications, feature, bins=10):
    if bins == "quartiles":
        bins = compute_tiles(test_dataset, feature, tiles=4)

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
    # List of features
    feature_list = list(test_dataset.columns)[:-1]
    
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
    points_per_tile = int(len(dataset)/float(tiles))
    ordered_dataset = dataset.sort_values(by=feature)
    
    edge_values = [ordered_dataset[feature].min()]
    
    for index in range(tiles-1):
        edge_values += [ordered_dataset[feature][index*points_per_tile:(index+1)*points_per_tile-1].max()]
        
    edge_values += [ordered_dataset[feature].max()]
    
    return edge_values



def show_tiles_feature(dataset, feature, tiles = 4):
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
