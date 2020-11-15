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
import math
from tqdm import tqdm


class SpatialDistribution:
    """Creates a SpatialDistribution class through which we will
    information on the spatial distribution of the data at hand

    Args:
            data(Pandas dataframe):The data to be analyzed
            label_predicted(List or Pandas Series):The label predicted by the model
            label_true(List or pandas series):That holds the true labels
            type(str): To be implemente will be used to determine if we are dealing with
                       numeric, categorical or mixed data
    """

    def __init__(self, data, label_predicted, label_true, type=None):
        self._data = data
        self.type = type
        self.label_predicted = np.array(label_predicted)
        self.label_true = np.array(label_true)
        self._data_len = len(data)
        self.data_w_predlabel = self._append_prediction_label()
        self._col_names = list(data.columns)

        self._metrics_dict = dict(
            zip(
                ["overlap", "goodall2", "goodall3", "lin"],
                [self.overlap, self.goodall2, self.goodall3, self.lin],
            )  # Dictionary
        )
        self._counts_per_attribute = (
            self.__buildcounts()
        )  # dictionary of counts of occurances of attribute instances

        self.cat_data = None  # This will be use when we extend the module for numerical
        self.num_data = None  # plus categorical data types

        if self.type == "mix":
            self.num_data = self._data.select_dtypes(include="number")
            self.cat_data = self._data.select_dtype(exclude="number")

        pass

    def _append_prediction_label(self):
        """Creates and appends a column to the self._data that has a boolean variable that
        indicates if the data point was clasify correctly"""

        pred_status_label = pd.Series(
            [i == j for i, j in zip(self.label_predicted, self.label_true)]
        )

        return pd.concat(
            [self._data, pred_status_label.rename("correctly-predicted")], axis=1
        )

    def __buildcounts(self):
        """Builds a dictionary with the attributes as key that hold the counts of the occurrances
        of different values in the data
        """

        counts_dict = {}
        for attribute in self._col_names:
            counts_dict[attribute] = self._data[attribute].value_counts()
        return counts_dict

    def available_metrics(self):
        """Prints the available metrics """
        metrics = list(self._metrics_dict.keys())
        for metric in metrics:
            print(metric)

    def goodall2(self, dpoint1, dpoint2):
        """Computes the goodall2 similary measurement for categorical data, see paper by
        Varun, Shyam and Vipin in the bibliography carpet for reference
        Args:
            dpoint1(Pandas Series): Data point to compare
            dpoin2(Pandas Series): Data point to compare
        """

        intersection = dpoint1 == dpoint2
        common_attributes = list(dpoint1[intersection].index)
        goodall2_score = 0
        for attribute in common_attributes:
            attribute_score = 0
            common_value = dpoint1[attribute]
            counts = self._counts_per_attribute[attribute]
            for count in counts:
                if counts[common_value] < count:
                    attribute_score = self.goodall_frequency(count) + attribute_score

            goodall2_score = (1 - attribute_score) + goodall2_score

        return goodall2_score / len(dpoint1)

    def goodall3(self, dpoint1, dpoint2):
        """Computes the goodall3 similarity measurment for categorical data, see paper by
        Varun, Shyam, Vipin in the bibliography reference for reference
        Args:
            dpoint1(Pandas Series): Data point to compare
            dpoin2(Pandas Series): Data point to compare
        """

        d1_attributes = list(dpoint1.index)
        d2_attributes = list(dpoint2.index)
        goodall3_score = 0
        if d1_attributes == d2_attributes:
            for attribute in d1_attributes:
                attribute_score = 0
                value_point1 = dpoint1[attribute]
                value_point2 = dpoint2[attribute]
                if value_point1 == value_point2:
                    count = self._counts_per_attribute[attribute][value_point1]

                    attribute_score = self.goodall_frequency(count)

                    goodall3_score = (1 - attribute_score) + goodall3_score

            return goodall3_score / len(d1_attributes)
        else:
            # write exception
            pass

    def overlap(self, dpoint1, dpoint2):
        """Computes the overlap similarity measure for categorical data
        Args:
            dpoint1(Pandas Series): Data point to compare
            dpoin2(Pandas Series): Data point to compare
        """

        if list(dpoint1.index) == list(dpoint2.index):
            overlap_score = (dpoint1 == dpoint2).mean()
            return overlap_score
        else:
            # write exception

            pass

    def lin(self, dpoint1, dpoint2):
        """Computes the Lin similarity measurment for categorical data, see paper by
        Varun, Shyam, Vipin in the bibliography reference for reference
        Args:
            dpoint1(Pandas Series): Data point to compare
            dpoin2(Pandas Series): Data point to compare
        """
        lin = 0
        weight = self.lin_weight(dpoint1, dpoint2)
        d1_attributes = list(dpoint1.index)
        for attribute in d1_attributes:
            frequency_dpoint1 = self.empirical_frequency(dpoint1[attribute], attribute)
            frequency_dpoint2 = self.empirical_frequency(dpoint2[attribute], attribute)
            if dpoint1[attribute] == dpoint2[attribute]:
                lin = lin + 2 * np.log(frequency_dpoint1)
            else:
                lin = lin + 2 * np.log(frequency_dpoint1 + frequency_dpoint2)
        return weight * lin

    def lin_weight(self, dpoint1, dpoint2):
        """Computes the lin wiehgt as describe in the paper of Varun, Shyam and Vipin in the
        bibliography"""

        d1_attributes = list(dpoint1.index)
        weight_denominator = 0
        for attribute in d1_attributes:
            frequency_dpoint1 = self.empirical_frequency(dpoint1[attribute], attribute)
            frequency_dpoint2 = self.empirical_frequency(dpoint2[attribute], attribute)
            weight_denominator = (
                weight_denominator
                + np.log(frequency_dpoint1)
                + np.log(frequency_dpoint2)
            )

        return 1 / weight_denominator

    def empirical_frequency(self, value, attribute):
        counts = self._counts_per_attribute[attribute][value]
        return counts / self._data_len

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
             dpoint(Pandas Series): Pandas Series data point to work with
             metric (str): similarity metric used to compute the distance
             distance_sample(numeric): Number that if from 0 to 1  indicates the percentage
                              from the total data that should be used to
                              estimate the average distance and if greater indicates the precise number
                              of samples to be take"""

        if distance_sample < 1:
            distance_sample = round(distance_sample * self._data_len)
        elif distance_sample > self._data_len:
            distance_sample = self._data_len
        else:
            distance_sample = self._data_len

        metric = self._metrics_dict[str(metric)]
        sampled_indexes = random.sample(range(self._data_len), distance_sample)

        distance = 0
        for i in sampled_indexes:
            distance = distance + metric(dpoint, self._data.iloc[i])

        return distance / (distance_sample)

    def array_of_distance(self, dpoint, metric, k):
        """Creates a list that for a point holds (in soarted order) the next k-nearest point
        to in  given a metric (ie the highest the position in the list the furtherst it is
        from the point )

        Args:
            dpoint1(Pandas Series): Data point from which we'll get the nearest neighbours
            metric(str): The metric use
            k(int): The number of points
        """
        metric = self._metrics_dict[metric]
        distance_array = []
        for i in range(0, self._data_len):
            next_point = self._data.iloc[i]
            distance_new = metric(dpoint, next_point)
            distance_array.append((distance_new, i))

        distance_array.sort(key=lambda x: x[0])

        return distance_array

    def plot_knearest_points(self, dpoint, metric1, metric2, metric3, k):
        """Given a pandas Series data point computes and visualize the k-nearest points in metric1
        and how this same point projects in metric2 and metric3 plotting everything in a graph
        where metric1 will be in the y-axis and the points in metric2 and metric3 will be projected in quadrant
        I and IV correspondingly

        Args:
             dpoin1 (Pandas Series): Data point from which we'll find the k nearest neighbours
             metric1 (str): Name of the similarity metric used to find the k nearest neighbours
             metric2 (str): Name of a similarity metric to project the knn in the I quadrant
             metric3 (str): Name of a similarity metric to project the knn in the IV quadrant
             k (int): Number of nearest neighbours to find
        """

        metrics = [metric2, metric3]
        main_metric = metric1
        k_nearest_per_metric = self.array_of_distance(dpoint, main_metric, k)

        sns.set()
        fig, ax = plt.subplots(1, 2)
        fig.subplots_adjust(wspace=0.25)
        metric1_to_point = [tuple[0] for tuple in k_nearest_per_metric][:k]
        index_of_neighbour = [tuple[1] for tuple in k_nearest_per_metric][:k]

        kpoints_other_space = []
        for metric in metrics:
            metric = self._metrics_dict[metric]

            metric_distance = [
                metric(dpoint, self._data.iloc[i]) for i in index_of_neighbour
            ][:k]
            kpoints_other_space.append(metric_distance)

        ax[0].set_xlim(kpoints_other_space[0][-1] * 1.05, 0)
        ax[0].set_xlabel(metrics[0] + " distance")
        ax[1].set_xlabel(metrics[1] + " distance")
        ax[0].yaxis.set_ticks_position("right")
        ax[1].set_ylabel(main_metric + " distance")
        sns.scatterplot(
            kpoints_other_space[0],
            metric1_to_point,
            ax=ax[0],
            s=10,
            hue=index_of_neighbour,
            alpha=0.3,
            legend=False,
        )
        sns.scatterplot(
            kpoints_other_space[1],
            metric1_to_point,
            ax=ax[1],
            s=10,
            hue=index_of_neighbour,
            alpha=0.4,
            legend=False,
        )

        plt.show()

    def plot_distance_histogram(
        self,
        metric,
        histo_sample=500,
        distance_sample=0.005,
        mdistance_sample=0.05,
        bar_width=0.01,
        ax=None,
    ):
        """Plots an histogram that approximates the distribution of the distance of the point relative to every
        other, it does it by taking random samples from the data of size defined by the user
        Args:
            metric (str): distance metric used to compute the distance
            histo_sample(int): number of points that are going to be used to create the histogram
            distance_sample(float/int): Percentage of the correctly classified  data that should be used for each point sampled for
                                        the histogram to compute it's approximate distance to every other point. if less or equal to 1 it
                                        represents a percentage if greater than 1 it represents the exact number of sampled points

            mdistance_sample(float/int): Percentage of the  misclassified  data that should be used for each point sampled for
                                        the histogram to compute it's approximate distance to every other point. if less or equal to 1 it
                                        represents a percentage if greater than 1 it represents the exact number of sampled points
            bar_width(float): Width of the histogram bars
            ax (Matplotlib Axis object): Axis to plot the graph
        """

        if histo_sample < 1:
            histo_sample = round(histo_sample * self._data_len)
        elif histo_sample > self._data_len:
            histo_sample = self._data_len

        distance_array = np.empty(histo_sample)
        sampled_indexes = random.sample(range(self._data_len), histo_sample)

        print("Processing correctly classified datapoints")
        j = 0
        for i in tqdm(sampled_indexes):
            distance_array[j] = self.distance_to_data(
                self._data.iloc[i], metric, distance_sample
            )
            j = j + 1

        # Processing the misclasfied points
        condition = ~self.data_w_predlabel["correctly-predicted"]
        misclass_indexes = self.data_w_predlabel.index[condition].tolist()
        misclass_distance_array = np.empty(len(misclass_indexes))
        print("Processing misclasfied datapoints")
        for i in tqdm(range(0, len(misclass_indexes))):
            index = misclass_indexes[i]
            misclass_distance_array[i] = self.distance_to_data(
                self._data.iloc[index], metric, mdistance_sample
            )

        sns.set()
        all_data_std = np.std(distance_array)
        all_data_mean = np.mean(distance_array)
        m_data_std = np.std(misclass_distance_array)
        m_data_mean = np.mean(misclass_distance_array)
        label_all_data = (
            "All points: "
            + "std= "
            + str(round(all_data_std, 3))
            + " mean= "
            + str(round(all_data_mean, 3))
        )
        label_m_data = (
            "misclassified: "
            + "std= "
            + str(round(m_data_std, 3))
            + " mean= "
            + str(round(m_data_mean, 3))
        )

        n = math.ceil((distance_array.max() - distance_array.min()) / bar_width)
        show = False
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title("Data distance histogram")
            show = True
        ax.grid()
        ax.set_alpha(0.5)
        ax.set_xlabel(metric + " distance distribution")
        ax.set_ylabel("Percentage of points (%)")
        sns.distplot(
            distance_array, norm_hist=True, bins=n, label=label_all_data, ax=ax
        )
        sns.distplot(
            misclass_distance_array, bins=n, label=label_m_data, ax=ax, norm_hist=True
        )
        ax.legend()

        if show is True:
            plt.show()
        pass

    def plot_full_histogram_report(
        self,
        histo_sample=2000,
        distance_sample=0.005,
        mdistance_sample=0.05,
        bar_width=0.01,
    ):
        """Plots the histograms and the kernel density estimation for the
        distribution in all the available metrics of the average distance that a point
        has to every other point
        Args:
             dpoint (Pandas Series): Pandas Series data point to work with
             metric (str): similarity metric used to compute the distance
             distance_sample (numeric): number that if between 0 to 1 that indicates the percentage
                              from the total data that should be used to
                              estimate the average distance for correctly classified
                              points if greater indicates the number of samples used to computed
                              the average distance of a given point to every other
             mdistance_sample (numeric): The same as above but for points that were misclasfied
             bar_width (float): The width of the histograms bar
        """

        col_number = 2
        row_number = 2
        fig, axis = plt.subplots(row_number, col_number)
        metrics = list(self._metrics_dict.keys())
        fig.suptitle("Data distance histograms")
        fig.subplots_adjust(hspace=0.25)

        for i in range(0, row_number):
            for j in range(0, col_number):
                print(metrics[(2 * i) + j] + ": ")
                self.plot_distance_histogram(
                    metrics[(2 * i) + j],
                    distance_sample=0.001,
                    mdistance_sample=0.05,
                    histo_sample=500,
                    ax=axis[i][j],
                )

        plt.show()
        return

    def plot_distance_misclasified(
        self, metric1, metric2, distance_sample=0.01, scatter_sample=1
    ):
        """Plots a scatter point of the average distance of the misclasfied points to every
        other point
        in 2 metrics using the x and y axis correspondingly
        Args:
            metric1 (str): similarity metric used in the x axis
            metric2 (str): similarity metric used in the y axis
            distance_sample (numeric): Number that if from 0 to 1  indicates the percentage
                             from the total data that should be used to
                             estimate the average distance and if greater indicates the precise number
                             of samples to be take
            scatter_sample(numeric): Number that if from 0 to 1 indicates the percentage from the total
                                     misclassified points that should be sample to plot if greater indicates
                                     the exact size of the sample rather than the percentage
        """
        data = self._data
        data_indexes = self.data_w_predlabel.index
        condition = ~self.data_w_predlabel["correctly-predicted"]
        missclass_indexes = data_indexes[condition].tolist()

        distance_array_metric1, distance_array_metric2 = (
            np.empty(len(missclass_indexes)),
            np.empty(len(missclass_indexes)),
        )
        if scatter_sample == 1:
            distance_index = 0
            for index in tqdm(missclass_indexes):
                distance_array_metric1[distance_index] = self.distance_to_data(
                    data.iloc[index], metric1, distance_sample=distance_sample
                )
                distance_array_metric2[distance_index] = self.distance_to_data(
                    data.iloc[index], metric2, distance_sample=distance_sample
                )
                distance_index = distance_index + 1
                continue

            sns.set()
            fig, ax = plt.subplots()
            sns.scatterplot(
                distance_array_metric1,
                distance_array_metric2,
            )
            ax.set_xlabel(metric1 + " average distance to other points")
            ax.set_ylabel(metric2 + " average distance to other points")
            ax.set_title("Data distance")
            plt.show()
            return fig, ax, distance_array_metric1, distance_array_metric2

    def plot_distance_scatterplot(
        self, metric1, metric2, scatter_sample=0.1, distance_sample=0.001
    ):
        """Produces a scatter plot of the distance of every point to every other
        using to different metrics as axises
        Args:
                metric1(str): metric to be used in the x axis
                metric2(str): metric to be used in the y axis
                scatter_sample(numeric): If less than  or equal 1 it represents the
                percentage of the data to be plotted if greater than 1 it represents the exact number of sampled_indexes
                distance_sample(numeric):  If less than  or equal 1 it represents the
                percentage of the data to be used in the estimation of the distance of the average distance of the points
                to every other point. If greater than 1 it represents the exact size of the sample"""

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
        for index in tqdm(sampled_indexes):
            distance_array_metric1[distance_index] = self.distance_to_data(
                data.iloc[index], metric1, distance_sample=distance_sample
            )
            distance_array_metric2[distance_index] = self.distance_to_data(
                data.iloc[index], metric2, distance_sample=distance_sample
            )
            distance_index = distance_index + 1

        sns.set()
        fig, ax = plt.subplots()
        data_w_predlabel = self._append_prediction_label()
        label = np.asarray(data_w_predlabel["correctly-predicted"][sampled_indexes])
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
