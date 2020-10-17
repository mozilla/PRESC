# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:47:47 2020

@author: castromi
"""


from matplotlib import pyplot as plt
import numpy as np
import random
import seaborn as sns
import pandas as pd


class SpatialDistribution:
    """Creates a SpatialDistribution class through which we will
    information on the spatial distribution of the data at hand

    Args:
            data: pandas dataframe of the data to be analyzed
            label_predicted: List or pandas series that holds the label predicted
                             by the model
            label_true: List or pandas series that holds the true labels
            model:Currently not implemented and set to None, in the future it might ask for the specific
                    sklearn trained model for more functionalities
    """

    def __init__(self, data, label_predicted, label_true, model=None):

        self._data = data
        self._data_len = len(data)
        self._col_names = list(data.columns)
        self._model = model
        self._metrics_dict = dict(
            zip(["overlap", "goodall2"], [self.overlap, self.goodall2])  # Dictionary
        )
        self._counts_per_attribute = (
            self.__buildcounts()
        )  # dictionary of counts of occurances of attribute instances
        self._append_prediction_label()

        pass

    def _append_prediction_label(self):
        """Creates and appends a column to the self._data that has a boolean variable that
        indicates if the data point was clasify correctly"""

        pred_status_label = pd.Series(self.label_predicted) == pd.Series(
            self.label_true
        )
        pd.concat([self._data, pred_status_label.rename("correctly-predicted")], axis=1)

    def __buildcounts(self):
        """Builds a dictionary with the attributes as key that hold the counts of the occurrances
        of different values in the data
        """
        counts_dict = {}
        for attribute in self._col_names:
            counts_dict[attribute] = self._data[attribute].value_counts()
        return counts_dict

    def goodall2(self, dpoint1, dpoint2):
        """Computes the goodall2 similary measurement for categorical data, see paper by
        Varun, Shyam and Vipin in the bibliography carpet for reference"""

        d1_attributes = list(dpoint1.index)
        d2_attributes = list(dpoint2.index)

        if d1_attributes == d2_attributes:
            intersection = dpoint1 == dpoint2
            common_attributes = list(dpoint1[intersection].index)
            goodall2_score = 0
            for attribute in common_attributes:
                attribute_score = 0
                counts = self._counts_per_attribute[attribute]

                for count in counts:
                    attribute_score = self.goodall_frequency(count) + attribute_score
                    pass

                goodall2_score = (1 - attribute_score) + goodall2_score

            return goodall2_score / len(d1_attributes)
        else:
            # Raise exeption points do not have the same attributes
            pass
        pass

    def goodall3(self, dpoint1, dpoint2):
        """Computes the goodall3 similarity measurment for categorical data, see paper by
        Varun, Shyam, Vipin in the bibliography reference for reference"""

        d1_attributes = list(dpoint1.index)
        d2_attributes = list(dpoint2.index)
        goodall3_score = 0
        if d1_attributes == d2_attributes:
            for attribute in d1_attributes:
                attribute_score = 0
                value_point1 = dpoint1[attribute]
                value_point2 = dpoint2[attribute]
                if value_point1 == value_point2:
                    count = self._counts_per_attribute[attribute]
                    attribute_score = self.modified_frequency(count)
                    goodall3_score = (1 - attribute_score) + goodall3_score

            return goodall3_score / len(d1_attributes)
        else:
            # raise exception
            pass

        pass

    def overlap(self, dpoint1, dpoint2):
        """Computes the overlap similarity measure for categorical data"""

        if list(dpoint1.index) == list(dpoint2.index):
            overlap_score = len(set(dpoint1) & set(dpoint2)) / len(dpoint1)
            return overlap_score
        else:
            # write exception

            pass

    def goodall_frequency(self, counts):
        """Computes a probability estimate of an attribute required to compute
        goodall similarity measures"""
        estimated_freq = (counts * (counts - 1)) / (
            (self._data_len) * (self._data_len - 1)
        )
        return estimated_freq

    def distance_to_data(self, dpoint, metric, distance_sample=0.013):
        """Computes an estimate of the average distance of a data point to every
        other data points by taking a random sample of a given size and avereging
        the result
        Args:
             dpoint: data point to work with
             metric: string, similarity metric used to compute the distance
             distance_sample: float for 0 to 1 that indicates the percentage
                              from the total data that should be used to
                              estimate the average distance"""

        if distance_sample < 1:
            distance_sample = round(distance_sample * self._data_len)
        elif distance_sample > self._data_len:
            distance_sample = self._data_len

        metric = self.metrics_dict[str(metric)]
        sampled_indexes = random.sample(range(self._data_len), distance_sample)

        distance = 0
        for i in sampled_indexes:
            distance = distance + metric(dpoint, self._data.iloc[i])

        return distance / (distance_sample)

    def plot_distance_histogram(self, metric, histo_sample=100, distance_sample=0.0073):
        """Plots an histogram that approximates the distribution of the distance of the point relative to every
        other, it does it by taking random samples from the data of size defined by the user
        Args:
            Metric: distance metric used to compute the distance
            histo_sample: number of points that are going to be used to create the histogram
            distance_sample: percentage of the data that should be used for each point sampled for
                              the histogram to compute it's approximate distance to every other point"""
        if histo_sample < 1:
            histo_sample = round(histo_sample * self._data_len)
        elif histo_sample > self._data_len:
            histo_sample = self._data_len

        distance_array = np.empty(histo_sample)
        sampled_indexes = random.sample(range(self._data_len), histo_sample)

        j = 0
        for i in sampled_indexes:
            distance_array[j] = self.distance_to_data(
                self._data.iloc[i], metric, distance_sample
            )
            print("Processing data :", round(100 * j / histo_sample, 5), "%", end="\r")
            j = j + 1
            continue

        sns.set()
        fig, ax = plt.subplots()
        ax.set_xlabel(metric + " distance distribution")
        ax.set_ylabel("counts")
        ax.set_title("Data distance histogram")
        ax.hist(distance_array, bins=20)
        plt.show()

    def plot_distance_scatterplot(
        self, metric1, metric2, scatter_sample=0.1, distance_sample=0.001
    ):
        """Produces a scatter plot of the distance of every point to every other
        using to different metrics as axises
        Args:
                metric1: metric to be used in the x axis
                metric2: metric to be used in the y axis
                scatter_sample: percentage of the data to be plotted
                distance_sample: percentage of the data to be used in the
                                distance estimation of every point in the
                                scattere plot"""

        data = self._data
        distance_array_metric1, distance_array_metric2 = (
            np.empty(scatter_sample),
            np.empty(scatter_sample),
        )

        if scatter_sample < 1:
            scatter_sample = round(scatter_sample * self._data_len)
        elif scatter_sample > self._data_len:
            scatter_sample = self._data_len

        sampled_indexes = random.sample(range(len(data)), scatter_sample)

        distance_index = 0
        for index in sampled_indexes:
            distance_array_metric1[distance_index] = self.distance_to_data(
                data.iloc[index], metric1, distance_sample=distance_sample
            )
            distance_array_metric2[distance_index] = self.distance_to_data(
                data.iloc[index], metric2, distance_sample=distance_sample
            )
            distance_index = distance_index + 1
            print(
                "Processing data :",
                round(100 * distance_index / scatter_sample, 5),
                "%",
                end="\r",
            )
            continue

        sns.set()
        fig, ax = plt.subplots()
        label = np.asarray(data.iloc[sampled_indexes]["correctly-predicted"])
        sns.scatterplot(
            distance_array_metric1,
            distance_array_metric2,
            size=label,
            sizes=(10, 80),
            hue=label,
        )
        ax.set_xlabel(metric1 + " average distance to other points")
        ax.set_ylabel(metric2 + " average distance to other points")
        ax.set_title("Data distance")
        plt.show()
        return fig, ax, distance_array_metric1, distance_array_metric2

    pass
