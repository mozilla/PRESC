from presc.evaluations.utils import get_bins, is_discrete

from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set here temporarily
METRIC = accuracy_score

DEFAULT_CONFIG = {"num_bins": 10, "quantile": False, "plot_width_fraction": 1.0}


def compute_conditional_metric(
    grouping_col, true_labs, pred_labs, metric, as_categorical=False, config=None
):
    """Compute metric values conditional on the grouping column.

    The metric is computed within unique values of the grouping column
    (categorical) or within bins partitioning its range (continuous).

    grouping_col: Series defining a grouping for the metric computation
    true_labs: Series of true labels for a test dataset
    pred_labs: Series of labels predicted by a model for a test dataset
    metric: the evaluation metric to compute across the groupings. This should be
        a function f(y_true, y_pred) which accepts Series of true and
        predicted labels.
    as_categorical: should the grouping column be treated as categorical, ie. binned
        on its unique values? If it is not numeric, this param is ignored.
    config: optional dict of config options to override the defaults

    Returns a `ConditionalMetricResult` instance.
    """
    _config = dict(DEFAULT_CONFIG)
    if config:
        _config.update(config)

    y_vals = DataFrame({"y_true": true_labs, "y_pred": pred_labs})
    if is_discrete(grouping_col):
        as_categorical = True
    if as_categorical:
        grouping = grouping_col
        bins = grouping.unique()
    else:
        grouping, bins = get_bins(
            grouping_col,
            num_bins=_config["num_bins"],
            quantile=_config["quantile"],
        )
    binned_metric_vals = y_vals.groupby(grouping).apply(
        lambda gp: metric(gp["y_true"], gp["y_pred"])
    )

    return ConditionalMetricResult(
        vals=binned_metric_vals,
        bins=Series(bins),
        categorical=as_categorical,
    )


class ConditionalMetricResult:
    """Result of the conditional metric evaluation for a single grouping.

    vals: a Series listing the computation result for each bin
    bins: a Series listing the bin endpoints. If the feature was treated as
        numeric, this will have length `len(vals)+1`, otherwise `len(vals)`.
    categorical: was the feature treated as categorical?
    """

    def __init__(self, vals, bins, categorical, config=None):
        self.vals = vals
        self.bins = bins
        self.categorical = categorical

    def display_result(self, xlab, ylab, config=None):
        """Display the evaluation result for the given grouping and metric.

        xlab: label to display on the x-axis
        ylab: label to display on the y-axis
        config: optional dict of config options to override the defaults
        """
        _config = dict(DEFAULT_CONFIG)
        if config:
            _config.update(config)

        if self.categorical:
            result_edges = self.bins.astype("str")
            alignment = "center"
            width_interval = 1
        else:
            result_edges = self.bins[:-1]
            alignment = "edge"
            # First element will be NaN.
            bin_widths = self.bins.diff()[1:]
            width_interval = bin_widths * _config["plot_width_fraction"]

        plt.ylim(0, 1)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.bar(
            result_edges,
            self.vals,
            width=width_interval,
            bottom=None,
            align=alignment,
            edgecolor="white",
            linewidth=2,
        )
        plt.show(block=False)


class ConditionalMetric:
    """Computation of confusion-based metrics across subsets of a test dataset.

    model: the ClassificationModel to run the evaluation for
    test_dataset: a Dataset to use for evaluation.
    config: optional dict of config options to override the defaults
    """

    def __init__(self, model, test_dataset, config=None):
        _config = dict(DEFAULT_CONFIG)
        if config:
            _config.update(config)
        self._config = _config
        self._model = model
        self._test_dataset = test_dataset
        self._test_pred = self._model.predict_labels(test_dataset)

    def compute_for_column(self, colname, metric, as_categorical=False):
        """Compute the evaluation for the given dataset column.

        The metric is computed within unique values of the specified column
        (if categorical) or within bins partitioning its range (if continuous).

        colname: a column in the dataset to partition on
        metric: the evaluation metric to compute across the partitions. This should be
            a function f(y_true, y_pred) which accepts Series of true and
            predicted labels.
        as_categorical: should the feature be treated as categorical, ie. binned
            on its unique values? If the feature is not numeric, this param is
            ignored.

        Returns a `ConditionalMetricResult` instance.
        """
        return compute_conditional_metric(
            grouping_col=self._test_dataset.df[colname],
            true_labs=self._test_dataset.labels,
            pred_labs=self._test_pred,
            metric=metric,
            as_categorical=as_categorical,
            config=self._config,
        )

    def display(self, colnames=None):
        """Displays the conditional metric result for each specified column.

        colnames: a list of column names to run the evaluation over, creating a plot
            for each. If not supplied, defaults to all feature columns.
        """
        if colnames is None:
            colnames = self._test_dataset.feature_names
        for colname in colnames:
            # TODO: don't hardcode the metric
            eval_result = self.compute_for_column(colname, metric=METRIC)
            eval_result.display_result(xlab=colname, ylab="Metric value")
