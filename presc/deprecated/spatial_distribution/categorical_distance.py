# -*- coding: utf-8 -*-

"""
Created on Fri Oct  9 11:47:47 2020
@author: castromi
"""


from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import seaborn as sns
import pandas as pd
import math
from tqdm import tqdm


class SpatialDistribution:

    """
    Creates a SpatialDistribution class through which we will
    information on the spatial distribution of the data at hand.

    Attributes
    ----------
    data : Pandas dataframe
        The data to be analyzed.
    label_predicted : List or Pandas Series
        The label predicted by the model.
    label_true : List or Pandas series
        That holds the true labels.
    type : str
        To be implemented - will be used to determine if we are dealing with numeric,
        categorical or mixed data.
    """

    def __init__(self, data, label_predicted, label_true, type=None):
        self._data = data
        self.cat_data = self._data.select_dtypes(exclude="number")
        self.num_data = self._data.select_dtypes(include="number")
        self.num_scaled_data = self.__build_scaled_data()
        self.type = type
        self.label_predicted = np.array(label_predicted)
        self.label_true = np.array(label_true)
        self._data_len = len(data)
        self.data_w_predlabel = self._append_prediction_label()
        self._all_col_names = list(data.columns)
        self._cat_col_names = list(self.cat_data.columns)
        self._num_col_names = list(self.num_data.columns)
        self._categoric_metrics_dict = dict(
            zip(
                ["overlap", "goodall2", "goodall3", "lin"],
                [self.overlap, self.goodall2, self.goodall3, self.lin],
            )  # Dictionary
        )
        self._numeric_metrics_dict = dict(
            zip(
                ["l2_norm", "l1_norm"],
                [self.l2_norm, self.l1_norm],
            )  # Dictionary
        )

        self._counts_per_attribute = (
            self.__buildcounts()
        )  # dictionary of counts of occurances of attribute instances

        pass

    def _append_prediction_label(self):
        """Creates and appends a column to the self._data that has a boolean variable that
        indicates if the data point was clasify correctly"""

        pred_status_label = pd.Series(
            [i == j for i, j in zip(self.label_predicted, self.label_true)],
            index=self._data.index,
        )

        return pd.concat(
            [self._data, pred_status_label.rename("correctly-predicted")], axis=1
        )

    def __buildcounts(self):
        """
        Builds a dictionary with the attributes as key that hold the counts of the occurrances
        of different values in the data
        """
        counts_dict = {}
        for attribute in self._cat_col_names:
            counts_dict[attribute] = self.cat_data[attribute].value_counts()
        return counts_dict

    def __build_scaled_data(self):
        """scales the numeric data returning a scaled data framed with the same column names """

        if not self.num_data.empty:
            num_scaled_data = StandardScaler().fit_transform(self.num_data.values)
            num_scaled_data_df = pd.DataFrame(
                num_scaled_data,
                index=self.num_data.index,
                columns=self.num_data.columns,
            )
            return num_scaled_data_df

    def get_available_metrics(self):
        """
        Prints the available metrics
        """
        cat_metrics = list(self._categoric_metrics_dict.keys())
        numeric_metrics = list(self._numeric_metrics_dict.keys())
        print("Categorical Metrics")
        for metric in cat_metrics:
            print(metric)
        print("Numeric Metrics")
        for metric in numeric_metrics:
            print(metric)

    def get_metric(self, metric):
        """
        Given the string representation of a metric returns the callable method
        and the type of the metric (either numerical or categorical) if the metric is not found
        because it's not implemented it raises value a error.

        Parameters
        ----------
        metric : str
            The name of the metric.
        """

        categorical_metric = self._categoric_metrics_dict.get(metric, None)

        if categorical_metric is None:
            numeric_metric = self._numeric_metrics_dict.get(
                metric, None
            )  # search in numeric dictionary
            if numeric_metric is None:
                raise ValueError("Specified metric is not implemented")
            return numeric_metric, "numeric"
        return categorical_metric, "categorical"

    def get_datapoint(self, index):
        """
        Returns a data point.

        Parameters
        ----------
            index : int
                Index of the datapoint to be returned.
        """
        return self._data.iloc[index]

    def get_data_len(self):
        return self._data_len

    def l2_norm(self, dpoint1, dpoint2):
        dpoint1 = np.array(self.num_scaled_data.loc[dpoint1.name])
        dpoint2 = np.array(self.num_scaled_data.loc[dpoint2.name])
        return np.linalg.norm(dpoint1 - dpoint2, ord=2)

    def l1_norm(self, dpoint1, dpoint2):
        dpoint1 = np.array(self.num_scaled_data.loc[dpoint1.name])
        dpoint2 = np.array(self.num_scaled_data.loc[dpoint2.name])
        return np.linalg.norm(dpoint1 - dpoint2, ord=1)

    def goodall2(self, dpoint1, dpoint2):
        """
        Computes the goodall2 similary measurement for categorical data, see paper by
        Varun, Shyam and Vipin in the bibliography carpet for reference.

        Parameters
        ----------
        dpoint1 : Pandas Series
            Data point to compare.
        dpoin2 : Pandas Series
            Data point to compare.
        """
        dpoint1 = dpoint1[self._cat_col_names]
        dpoint2 = dpoint2[self._cat_col_names]

        if len(list(dpoint1.index)) == 0:
            raise ValueError("The points don't have categorical attributes to compare")

        intersection = dpoint1 == dpoint2
        common_attributes = list(
            dpoint1[intersection].index
        )  # get the name of the features that match
        goodall2_score = 0
        for (
            attribute
        ) in common_attributes:  # for all the attribute categories they have in common
            attribute_score = 0
            common_value = dpoint1[
                attribute
            ]  # get the value of the attribute they both take
            counts = self._counts_per_attribute[
                attribute
            ]  # get a Series with the counts of how many times each of all the other possible values of that attribute occur
            for (
                count
            ) in (
                counts
            ):  # for the each of the attribute values counts of this common feature
                if count < counts[common_value]:
                    attribute_score = (
                        self.goodall_frequency(count) + attribute_score
                    )  # increase the attribute score using this count
            goodall2_score = (1 - attribute_score) + goodall2_score

        return goodall2_score / len(dpoint1)

    def goodall3(self, dpoint1, dpoint2):
        """
        Computes the goodall3 similarity measurment for categorical data, see paper by
        Varun, Shyam, Vipin in the bibliography reference for reference.

        Parameters
        ----------
        dpoint1 : Pandas Series
            Data point to compare.
        dpoin2 : Pandas Series
            Data point to compare.
        """
        dpoint1 = dpoint1[self._cat_col_names]
        dpoint2 = dpoint2[self._cat_col_names]

        d1_attributes = list(dpoint1.index)
        d2_attributes = list(dpoint2.index)
        goodall3_score = 0
        if d1_attributes != d2_attributes:
            raise ValueError("The points have different columns ")
        if len(d1_attributes) == 0:
            raise ValueError("The points don't have categorical attributes to compare")

        for attribute in d1_attributes:
            attribute_score = 0
            value_point1 = dpoint1[attribute]
            value_point2 = dpoint2[attribute]
            if value_point1 == value_point2:

                count = self._counts_per_attribute[attribute][value_point1]
                attribute_score = self.goodall_frequency(count)
                goodall3_score = (1 - attribute_score) + goodall3_score
        return goodall3_score / len(d1_attributes)

    def overlap(self, dpoint1, dpoint2):
        """
        Computes the overlap similarity measure for categorical data.

        Parameters
        ----------
        dpoint1 : Pandas Series
            Data point to compare.
        dpoin2 : Pandas Series
            Data point to compare.
        """
        dpoint1 = dpoint1[self._cat_col_names]
        dpoint2 = dpoint2[self._cat_col_names]

        if list(dpoint1.index) == list(dpoint2.index):
            overlap_score = (dpoint1 == dpoint2).mean()
            return overlap_score
        else:
            raise ValueError("The Panda Series have different columns ")

    def lin(self, dpoint1, dpoint2):
        """
        Computes the Lin similarity measurment for categorical data, see paper by
        Varun, Shyam, Vipin in the bibliography reference for reference.

        Parameters
        ----------
        dpoint1 : Pandas Series
            Data point to compare.
        dpoin2 : Pandas Series
            Data point to compare.
        """
        dpoint1 = dpoint1[self._cat_col_names]
        dpoint2 = dpoint2[self._cat_col_names]
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
        """
        Computes the lin weight as describe in the paper of Varun, Shyam and Vipin in the
        bibliography.
        """

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
        """
        Computes a probability estimate of an attribute required to compute
        goodall similarity measures.
        """
        estimated_freq = (counts * (counts - 1)) / (
            (self._data_len) * (self._data_len - 1)
        )
        return estimated_freq

    def distance_to_data(self, dpoint, metric, distance_sample=0.013):
        """
        Computes an estimate of the average distance of a data point to every
        other data point by taking a random sample of a given size and avereging
        the result.

        Parameters
        ----------
        dpoint : Pandas Series
            Pandas Series data point to work with.
        metric : str
            Similarity metric used to compute the distance.
        distance_sample : numeric
            Number that if from 0 to 1  indicates the percentage from the total data that
            should be used to estimate the average distance and if greater indicates the
            precise number of samples to be taken.

        Returns
        -------
        float
            Estimate of average distance from a point to the data.
        """

        metric, type = self.get_metric(metric)
        if type == "categorical":
            data = self.cat_data
            dpoint = data.loc[dpoint.name]
        else:
            data = self.num_scaled_data
            dpoint = data.loc[dpoint.name]

        if distance_sample <= 1:
            distance_sample = round(distance_sample * self._data_len)
            if distance_sample == 0:
                distance_sample = 1
        elif distance_sample > self._data_len:
            distance_sample = self._data_len
        else:
            distance_sample = self._data_len

        sampled_indexes = random.sample(list(data.index), distance_sample)

        distance = 0
        for i in sampled_indexes:
            distance = distance + metric(dpoint, data.loc[i])

        return distance / (distance_sample)

    def array_of_distance(
        self, dpoint, metric, misclass_only=False, correct_only=False
    ):
        """
        Creates a list of tuples (distance,index) that holds the distances of all
        the points to the specified dpoint in a given metric and their position (index) in the dataset.
        The list of tuples is soarted based on the distance in increasing order of distance.

        Parameters
        ----------
        dpoint : Pandas Series
            Data point from which we'll get the distances to the other points.
        metric : str
            The metric to use.
        missclass_only : bool
            Default to False, that if True the list only contains the tuples of (distance,index)
            to misclassified points.
        correct_only : bool
            Default to False, that if True the list only contains the tuples of
            (distance,index) to misclasfied points.

        Returns
        -------
        list
            List of tuples
        """
        metric, type = self.get_metric(metric)

        if type == "categorical":
            data = self.cat_data
            dpoint = data.loc[dpoint.name]  # get only the categorical entries

        else:
            data = self.num_data
            dpoint = data.loc[dpoint.name]  # get only the numerical entries

        distance_misclas_array = []
        distance_correct_only = []
        distance_array = []
        for i in range(0, self._data_len):
            nxt_point = data.iloc[
                i
            ]  # you are getting nxt_point from the data filtered with only numeric or categorical columns
            distance_new = metric(dpoint, nxt_point)
            distance_array.append((distance_new, i))
            nxt_point_correctly_predicted = self.data_w_predlabel.iloc[
                i
            ].loc[  # usign the index to prevent conflict
                "correctly-predicted"
            ]
            if nxt_point_correctly_predicted:
                distance_correct_only.append((distance_new, i))
            elif not nxt_point_correctly_predicted:
                distance_misclas_array.append((distance_new, i))

        if misclass_only and correct_only:
            raise ValueError("misclass_only and correct_only can't be both True")

        if misclass_only:
            distance_misclas_array.sort(key=lambda x: x[0])
            return distance_misclas_array
        if correct_only:
            distance_correct_only.sort(key=lambda x: x[0])
            return distance_correct_only

        distance_array.sort(key=lambda x: x[0])
        return distance_array

    def plot_distance_to_point_histogram(
        self, dpoint, metric, other_metrics=None, bar_width=0.01, show=True
    ):
        """
        Function that given a metric and a dpoint plots a graphs of the histograms
        of the distribution of the distance from dpoint to every point that has been
        correctly classified and the distance from dpoint to every point that has been
        misclassified

        Parameters
        ----------
        metric : str
            Name of the similarity metric to be used.
        dpoint : Pandas Series
            Data point that will be used as reference to compute the histogram of relative distances.
        other_metrics : list of strings
            In case the user wants to use more than one metric to compute the distance the other
            metrics that will be added up are to be specified here.
        bar_width: numeric
            Bar width for the histogram.
        show : bool
            Controls if the graph is shown.
        """

        distance_to_correctly_clasified = np.array(
            [
                tuple[0]
                for tuple in self.array_of_distance(dpoint, metric, misclass_only=False)
            ]
        )
        distance_to_misclasified = np.array(
            [
                tuple[0]
                for tuple in self.array_of_distance(dpoint, metric, misclass_only=True)
            ]
        )
        x_label = "distance to points in: " + str(metric)
        if other_metrics is not None:
            for metrics in other_metrics:
                new_distance_to_correctly_clasified = np.array(
                    [
                        tuple[0]
                        for tuple in self.array_of_distance(
                            dpoint, metric, misclass_only=False
                        )
                    ]
                )
                new_distance_to_misclasified = np.array(
                    [
                        tuple[0]
                        for tuple in self.array_of_distance(
                            dpoint, metric, misclass_only=True
                        )
                    ]
                )
                distance_to_correctly_clasified = np.add(
                    distance_to_correctly_clasified, new_distance_to_correctly_clasified
                )
                distance_to_misclasified = np.add(
                    distance_to_misclasified, new_distance_to_misclasified
                )
                x_label = x_label + " + " + metrics

        sns.set()
        correct_data_std = np.std(distance_to_correctly_clasified)
        correct_data_mean = np.mean(distance_to_correctly_clasified)
        m_data_std = np.std(distance_to_misclasified)
        m_data_mean = np.mean(distance_to_misclasified)

        label_all_data = (
            "All points: "
            + "(std= "
            + str(round(correct_data_std, 3))
            + " ,μ= "
            + str(round(correct_data_mean, 3))
            + ")"
        )
        label_m_data = (
            "misclassified: "
            + "(std= "
            + str(round(m_data_std, 3))
            + " μ= "
            + str(round(m_data_mean, 3))
            + ")"
        )

        fig, ax = plt.subplots()
        ax.set_title(
            "Histogram of the distribution of distances from the given to the rest of the data"
        )
        ax.set_alpha(0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Percentage of points")
        bins_number_correct = math.ceil(
            (
                distance_to_correctly_clasified.max()
                - distance_to_correctly_clasified.min()
            )
            / bar_width
        )
        bins_number_misclass = math.ceil(
            (distance_to_misclasified.max() - distance_to_misclasified.min())
            / bar_width
        )

        sns.histplot(
            distance_to_correctly_clasified,
            bins=bins_number_correct,
            label=label_all_data,
            ax=ax,
            kde=True,
            stat="density",
        )
        sns.histplot(
            distance_to_misclasified,
            bins=bins_number_misclass,
            label=label_m_data,
            ax=ax,
            kde=True,
            stat="density",
            color="coral",
        )
        ax.legend()
        if show:
            plt.show(block=False)

    def plot_avg_distance_histogram(
        self,
        metric,
        histo_sample=0.2,
        distance_sample=0.005,
        mdistance_sample=0.05,
        bar_width=0.01,
        ax=None,
    ):
        """
        Plots an histogram that approximates the distribution of the average distance of point
        relative to every other, it does so by taking random samples from the data of size
        defined by the user.

        Parameters
        ----------
        metric : str
            Distance metric used to compute the distance.
        histo_sample : int
            Number of points that are going to be used to create the histogram.
        distance_sample : float/int
            Percentage of the correctly classified  data that should be used for each point sampled for
            the histogram to compute it's approximate distance to every other point. if less or equal to 1 it
            represents a percentage if greater than 1 it represents the exact number of sampled points.
        mdistance_sample : float/int
            Percentage of the  misclassified  data that should be used for each point sampled for
            the histogram to compute it's approximate distance to every other point. if less or equal to 1 it
            represents a percentage if greater than 1 it represents the exact number of sampled points.
        bar_width : float
            Width of the histogram bars.
        ax : Matplotlib Axis object
            Defaults to None, allows the user to specify a particular axis to plot the graph
            instead of creating a new one.
        """

        show = False
        if ax is None:
            sns.set()
            fig, ax = plt.subplots()
            ax.set_title("Histogram of Estimate average distance between data points")
            show = True

        true_condition = self.data_w_predlabel["correctly-predicted"]
        true_class_indexes = self.data_w_predlabel.index[true_condition].tolist()

        if histo_sample <= 1:
            histo_sample = round(histo_sample * len(true_class_indexes))
        elif histo_sample > len(true_class_indexes):
            histo_sample = len(true_class_indexes)

        true_condition = self.data_w_predlabel["correctly-predicted"]
        true_class_indexes = self.data_w_predlabel.index[true_condition].tolist()
        distance_array = np.empty(histo_sample)

        sampled_indexes = random.sample(true_class_indexes, histo_sample)

        print("Processing correctly classified datapoints in metric ", metric)
        j = 0
        for i in tqdm(sampled_indexes):
            distance_array[j] = self.distance_to_data(
                self._data.loc[i], metric, distance_sample
            )
            j = j + 1

        # Processing the misclasfied points
        false_condition = ~self.data_w_predlabel["correctly-predicted"]
        misclass_indexes = self.data_w_predlabel.index[false_condition].tolist()
        misclass_distance_array = np.empty(len(misclass_indexes))
        print("Processing misclasfied datapoints in metric ", metric)
        for i in tqdm(range(0, len(misclass_indexes))):
            index = misclass_indexes[i]
            misclass_distance_array[i] = self.distance_to_data(
                self._data.loc[index], metric, mdistance_sample
            )

        all_data_std = np.std(distance_array)
        all_data_mean = np.mean(distance_array)
        m_data_std = np.std(misclass_distance_array)
        m_data_mean = np.mean(misclass_distance_array)
        label_all_data = (
            "All points: "
            + "(std= "
            + str(round(all_data_std, 3))
            + " ,μ= "
            + str(round(all_data_mean, 3))
            + ")"
        )
        label_m_data = (
            "misclassified: "
            + "(std= "
            + str(round(m_data_std, 3))
            + " μ= "
            + str(round(m_data_mean, 3))
            + ")"
        )

        n_clasified = math.ceil(
            (distance_array.max() - distance_array.min()) / bar_width
        )
        n_misclasified = math.ceil(
            (misclass_distance_array.max() - misclass_distance_array.min()) / bar_width
        )
        # ax.grid()
        ax.set_alpha(0.5)
        ax.set_xlabel(metric + " distance distribution")
        ax.set_ylabel("Percentage of points (%)")
        sns.histplot(
            distance_array,
            kde=True,
            stat="density",
            bins=n_clasified,
            label=label_all_data,
            ax=ax,
        )
        sns.histplot(
            misclass_distance_array,
            kde=True,
            stat="density",
            bins=n_misclasified,
            label=label_m_data,
            ax=ax,
            color="coral",
        )
        ax.legend()

        if show:
            plt.show(block=False)
        pass

    def plot_full_cat_histogram_report(
        self,
        histo_sample=2000,
        distance_sample=0.005,
        mdistance_sample=0.05,
        bar_width=0.01,
    ):
        """
        Plots the histograms and the kernel density estimation for the
        distribution in all the available metrics of the average distance that a point
        has to every other point.

        Parameters
        ----------
        dpoint : Pandas Series
            Pandas Series data point to work with.
        metric : str
            Similarity metric used to compute the distance.
        distance_sample : numeric
            Number that if between 0 to 1 that indicates the percentage from the total data that
            should be used to estimate the average distance for correctly classified
            points if greater indicates the number of samples used to computed
            the average distance of a given point to every other.
        mdistance_sample : numeric
            The same as above but for points that were misclassified.
        bar_width : float
            The width of the histograms bar.
        """

        col_number = 2
        row_number = 2
        fig, axis = plt.subplots(row_number, col_number, figsize=(24, 16))
        metrics = list(self._categoric_metrics_dict.keys())
        fig.suptitle("Data distance histograms")
        fig.subplots_adjust(hspace=0.25)

        for i in range(0, row_number):
            for j in range(0, col_number):
                self.plot_avg_distance_histogram(
                    metrics[(2 * i) + j],
                    distance_sample=0.001,
                    mdistance_sample=0.05,
                    histo_sample=500,
                    ax=axis[i][j],
                )

        plt.show(block=False)
        return

    def plot_distance_misclassified(
        self, metric1, metric2, distance_sample=0.01, scatter_sample=1
    ):
        """
        Plots a scatterplot of the average distance of the misclassified points to every
        other point in 2 metrics using the x and y axis correspondingly

        Parameters
        ----------
        metric1 : str
            Similarity metric used in the x axis.
        metric2 : str
            Similarity metric used in the y axis.
        distance_sample : numeric
            Number that if from 0 to 1  indicates the percentage from the total data that
            should be used to estimate the average distance and if greater indicates the
            precise number of samples to be take.
        scatter_sample : numeric
            Number that if from 0 to 1 indicates the percentage from the total  misclassified
            points that should be sample to plot if greater indicates the exact size of the
            sample rather than the percentage.
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
                    data.loc[index], metric1, distance_sample=distance_sample
                )
                distance_array_metric2[distance_index] = self.distance_to_data(
                    data.loc[index], metric2, distance_sample=distance_sample
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
            ax.set_title("Misclassified points average distance to all data")
            plt.show(block=False)
            return fig, ax, distance_array_metric1, distance_array_metric2

    def plot_distance_scatterplot(
        self,
        metric1,
        metric2,
        scatter_sample=0.1,
        distance_sample=0.001,
    ):
        """
        Produces a scatter plot of the distance of every point to every other
        using to different metrics as axis.

        Parameters
        ----------
        metric1 : str
            Metric to be used in the x axis.
        metric2 : str
            Metric to be used in the y axis.
        scatter_sample : numeric
            If less than  or equal 1 it represents the percentage of the data to be
            plotted if greater than 1 it represents the exact number of sampled_indexes.
        distance_sample : numeric
            If less than  or equal 1 it represents the percentage of the data to be
            used in the estimation of the distance of the average distance of the points
            to every other point. If greater than 1 it represents the exact size of the sample.
        """

        data = self._data

        if scatter_sample <= 1:
            scatter_sample = round(scatter_sample * self._data_len)
        elif scatter_sample > self._data_len:
            scatter_sample = self._data_len
        distance_array_metric1, distance_array_metric2 = (
            np.empty(scatter_sample),
            np.empty(scatter_sample),
        )
        sampled_indexes = random.sample(list(self._data.index), scatter_sample)

        distance_index = 0
        for index in tqdm(sampled_indexes):
            distance_array_metric1[distance_index] = self.distance_to_data(
                data.loc[index], metric1, distance_sample=distance_sample
            )
            distance_array_metric2[distance_index] = self.distance_to_data(
                data.loc[index], metric2, distance_sample=distance_sample
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
            sizes=(20, 60),
            hue=label,
        )
        ax.set_xlabel(metric1 + " average distance to other points")
        ax.set_ylabel(metric2 + " average distance to other points")
        ax.set_title(
            "Scatterplot of estimated average distance of a data point to every other"
        )
        ax.legend(title="Corretly classified?")
        plt.show(block=False)
        return fig, ax
