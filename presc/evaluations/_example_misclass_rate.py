import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from presc.evaluations.base_evaluation import BaseEvaluation


def is_discrete(feature):
    """Returns True if the given feature should be considered discrete/categorical."""
    return is_bool_dtype(feature) or not is_numeric_dtype(feature)


class MisclassRateEvaluation(BaseEvaluation):
    """Sample implementation of an evaluation method based on misclass_rate.py."""

    _default_config = {"num_bins": 10, "show_sd": False, "width_fraction": 1.0}

    def __init__(self, model, config=None):
        super().__init__(model=model, config=config)

    def _misclass_rate_feature(self, feature_name, categorical=False):
        """Computes the misclassification rate for binned values of a feature."""
        test_feat = self._dataset.test_features[feature_name]
        misclass_feat = test_feat[self._model.test_misclassified]

        if categorical is False:
            # Histogram of all points
            total_histogram_counts, bins = np.histogram(
                test_feat, self._config["num_bins"]
            )
            # Histogram of misclassified points
            misclass_histogram_counts, bins = np.histogram(misclass_feat, bins)

        else:
            # Histogram of all points for categorical features
            total_histogram_counts = test_feat.value_counts().sort_index()

            # Histogram of misclassified points for categorical features
            correct_feat = test_feat[~self._model.test_misclassified]
            misclass_histogram_counts = total_histogram_counts.subtract(
                correct_feat.value_counts(), fill_value=0
            )

            bins = np.asarray(misclass_histogram_counts.index)
            misclass_histogram_counts = np.asarray(misclass_histogram_counts)
            total_histogram_counts = np.asarray(total_histogram_counts)

        # Compute misclassification rate

        # The standard deviation in a counting experiment is N^(1/2).
        # According to error propagation the error of a quotient X=M/N is:
        # ErrorX = X(ErrorM/M + ErrorN/N),
        # here, Error_rate = rate*(M^(-1/2)+N^(-1/2))

        misclass_rate_histogram = np.copy(misclass_histogram_counts)

        rate = []
        standard_deviation = []
        for index in range(len(total_histogram_counts)):
            if total_histogram_counts[index] != 0:
                index_rate = (
                    misclass_rate_histogram[index] / total_histogram_counts[index]
                )
                rate += [index_rate]
                if misclass_rate_histogram[index] != 0:
                    standard_deviation += [
                        index_rate
                        * (
                            total_histogram_counts[index] ** (-1.0 / 2)
                            + misclass_rate_histogram[index] ** (-1.0 / 2)
                        )
                    ]
                else:
                    standard_deviation += [float("nan")]
            else:
                rate += [float("nan")]
                standard_deviation += [float("nan")]
        misclass_rate_histogram = rate

        return bins, misclass_rate_histogram, standard_deviation

    def _show_misclass_rate_feature(self, feature_name):
        """Displays the misclassification rate for the values of a certain feature. """
        categorical = is_discrete(self._dataset.features[feature_name])
        result_edges, result_rate, result_sd = self._misclass_rate_feature(
            feature_name, categorical=categorical
        )
        if categorical is False:
            width = np.diff(result_edges)
            width_interval = [bin * self._config["width_fraction"] for bin in width]
            result_edges = result_edges[:-1]
            alignment = "edge"
        else:
            result_edges = [str(item) for item in result_edges]
            alignment = "center"
            width_interval = 1

        plt.ylim(0, 1)
        plt.xlabel(feature_name)
        plt.ylabel("Misclassification rate")
        if self._config["show_sd"]:
            plt.bar(
                result_edges,
                result_rate,
                yerr=result_sd,
                width=width_interval,
                bottom=None,
                align=alignment,
                edgecolor="white",
                linewidth=2,
            )
        else:
            plt.bar(
                result_edges,
                result_rate,
                width=width_interval,
                bottom=None,
                align=alignment,
                edgecolor="white",
                linewidth=2,
            )
        plt.show(block=False)

    def display(self):
        """Displays the misclassification rate for the values of each feature."""
        # Computes position of bin edges for quartiles or deciles
        for feat in self._dataset.feature_names:
            self._show_misclass_rate_feature(feat)
