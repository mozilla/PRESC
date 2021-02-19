from presc.evaluations.utils import is_discrete

from numpy import histogram, histogram_bin_edges
from pandas import Series, MultiIndex
import matplotlib.pyplot as plt

DEFAULT_CONFIG = {"binning": "fd", "common_bins": True, "plot_width_fraction": 1.0}


def _histogram_bin_labels(bin_edges):
    """Returns a Series of labels for histogram bins generated using
    `numpy.histogram_bin_edges`."""
    labs = [
        f"[{bin_edges[i]:g}, {bin_edges[i+1]:g})" for i in range(len(bin_edges) - 1)
    ]
    labs[-1] = labs[-1][:-1] + "]"
    return Series(labs)


def compute_conditional_distribution(
    data_col, true_labs, pred_labs, as_categorical=False, config=None
):
    """Compute a distributional summary.

    The metric is computed within unique values of the grouping column
    (categorical) or within bins partitioning its range (continuous).

    data_col: a column of data from a test dataset
    true_labs: Series of true labels for the test dataset
    pred_labs: Series of labels predicted by a model for the test dataset
    as_categorical: should the data column be treated as categorical, ie. binned
        on its unique values? If it is not numeric, this param is ignored.
    config: optional dict of config options to override the defaults

    Returns a `ConditionalMetricResult` instance.
    """
    _config = dict(DEFAULT_CONFIG)
    if config:
        _config.update(config)

    grouping = [true_labs, pred_labs]
    if is_discrete(data_col):
        as_categorical = True
    if as_categorical:
        grouping.append(data_col)
        distribs = data_col.groupby(grouping).size()
        if _config["common_bins"]:
            # Extend the index in each label group to include all data values
            data_vals = distribs.index.get_level_values(-1).unique()
            y_vals = distribs.index.droplevel(-1).unique()
            full_ind = MultiIndex.from_tuples(
                [(yt, yp, x) for yt, yp in y_vals.values for x in data_vals],
                names=distribs.index.names,
            )
            distribs = distribs.reindex(index=full_ind, fill_value=0)
            bin_edges = Series(data_vals)
        else:
            # Convert the innermost index level to a Series of bin edges.
            bin_edges = distribs.rename(None).reset_index(level=-1).iloc[:, 0]
    else:
        if _config["common_bins"]:
            bins = histogram_bin_edges(data_col, bins=_config["binning"])
        else:
            bins = _config["binning"]
        # distribs will be a series with values (<hist_values>, <bin_edges>)
        distribs = data_col.groupby(grouping).apply(lambda x: histogram(x, bins=bins))
        bin_edges = distribs.map(lambda x: x[1])
        bin_ind_tuples = []
        for y in distribs.index:
            bin_ind_tuples.extend(
                [(y[0], y[1], x) for x in _histogram_bin_labels(bin_edges.loc[y])]
            )
        index_with_bins = MultiIndex.from_tuples(
            bin_ind_tuples, names=distribs.index.names + [None]
        )
        distribs = Series(
            distribs.map(lambda x: x[0]).explode().values, index=index_with_bins
        )
        if _config["common_bins"]:
            # Retain the unique bin edges as an array
            bin_edges = Series(bin_edges.iloc[0])

    return ConditionalDistributionResult(
        vals=distribs,
        bins=Series(bin_edges),
        categorical=as_categorical,
    )


class ConditionalDistributionResult:
    """Result of the conditional distribution computation for a single column of data.

    vals: a Series listing the bin counts for each group, indexed by
        (<true_label>, <predicted_label>, <bin_label>).
    bins: a Series listing the bin endpoints. If `common_bins` is `False`,
        this should be indexed by (<true_label>, <predicted_label>) and list
        the endpoints for each group. If the data was treated as numeric, this
        will have length `len(vals)+1` (within each group), otherwise
        `len(vals)`.
    categorical: was the feature treated as categorical?
    """

    def __init__(self, vals, bins, categorical, config=None):
        self.vals = vals
        self.bins = bins
        self.categorical = categorical

    def display_result(self, xlab, config=None):
        """Display the distributions for the given data column.

        xlab: label to display on the x-axis
        ylab: label to display on the y-axis
        config: optional dict of config options to override the defaults
        """
        _config = dict(DEFAULT_CONFIG)
        if config:
            _config.update(config)

        for y_true, y_pred in self.vals.index.droplevel(-1).unique():
            counts = self.vals.loc[(y_true, y_pred)]
            if isinstance(self.bins.index, MultiIndex):
                bins = self.bins.loc[(y_true, y_pred)]
            else:
                bins = self.bins
            if self.categorical:
                plt.bar(
                    bins.astype("str"),
                    counts,
                )
            else:
                bins = bins.values
                plt.hist(
                    (bins[:-1] + bins[1:]) / 2,
                    bins=len(counts),
                    weights=counts,
                    range=(bins.min(), bins.max()),
                )
            plt.xlabel(xlab)
            plt.ylabel("Frequency")
            plt.title(f"Group: {y_true}_predicted_as_{y_pred}")

            plt.show(block=False)


class ConditionalDistribution:
    """Computation of data distributions conditional on prediction results.

    model: the ClassificationModel to run the evaluation for
    test_dataset: a Dataset to use for evaluation.
    config: optional dict of config options to override the defaults. Available options:
        `binning`: binning scheme to use for a numerical column, passed to `numpy.histogram`.
            Can be a fixed number of bins or a string indicating a binning scheme, default: "fd"
        `common_bins`: should the bins be computed over the entire column and shared
            across groups (`True`) or computed within each group (`False`), default: True
        `plot_width_fraction`: width of the bars relative to available space on the plot.
            Smaller means more space between the bars, default: 1.0
    """

    def __init__(self, model, test_dataset, config=None):
        _config = dict(DEFAULT_CONFIG)
        if config:
            _config.update(config)
        self._config = _config
        self._model = model
        self._test_dataset = test_dataset
        self._test_pred = self._model.predict_labels(test_dataset)

    def compute_for_column(self, colname, as_categorical=False):
        """Compute the evaluation for the given dataset column.

        colname: a column in the dataset to compute distributions for
        as_categorical: should the feature be treated as categorical, ie.
            binned on its unique values? If the feature is not numeric, this
            param is ignored.
        Returns a `ConditionalDistributionResult` instance.
        """
        return compute_conditional_distribution(
            data_col=self._test_dataset.df[colname],
            true_labs=self._test_dataset.labels,
            pred_labs=self._test_pred,
            as_categorical=as_categorical,
            config=self._config,
        )

    def display(self, colnames=None):
        """Computes and displays the conditional distribution result for each specified column.

        colnames: a list of column names to run the evaluation over, creating a plot
            for each. If not supplied, defaults to all feature columns.
        """
        if colnames is None:
            colnames = self._test_dataset.feature_names
        for colname in colnames:
            eval_result = self.compute_for_column(colname)
            eval_result.display_result(xlab=colname)
