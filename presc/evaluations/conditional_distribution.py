from presc.evaluations.utils import is_discrete
from presc.utils import include_exclude_list
from presc.configuration import PrescConfig
from presc import global_config

from numpy import histogram, histogram_bin_edges
from pandas import Series, MultiIndex
import matplotlib.pyplot as plt
from confuse import ConfigError


def _histogram_bin_labels(bin_edges):
    """Returns a Series of labels for histogram bins generated using
    `numpy.histogram_bin_edges`."""
    labs = [
        f"[{bin_edges[i]:g}, {bin_edges[i+1]:g})" for i in range(len(bin_edges) - 1)
    ]
    labs[-1] = labs[-1][:-1] + "]"
    return Series(labs)


def compute_conditional_distribution(
    data_col, true_labs, pred_labs, as_categorical=False, binning="fd", common_bins=True
):
    """Compute a distributional summary.

    The metric is computed within unique values of the grouping column
    (categorical) or within bins partitioning its range (continuous).

    Parameters
    ----------
    data_col :
        A column of data from a test dataset.
    true_labs : Series
        A series of true labels for the test dataset.
    pred_labs : Series
        A series of labels predicted by a model for the test dataset.
    as_categorical : bool
        Should the data column be treated as categorical, ie. binned
        on its unique values? If it is not numeric, this param is ignored.
    binning : str
        Binning scheme to use for a numerical column, passed to `numpy.histogram`.
        Can be a fixed number of bins or a string indicating a binning scheme
    common_bins : bool
        Should the bins be computed over the entire column and shared
        across groups (`True`) or computed within each group (`False`)

    Returns
    -------
    ConditionalDistributionResult
    """

    grouping = [true_labs, pred_labs]
    if is_discrete(data_col):
        as_categorical = True
    if as_categorical:
        grouping.append(data_col)
        distribs = data_col.groupby(grouping).size()
        if common_bins:
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
        if common_bins:
            bins = histogram_bin_edges(data_col, bins=binning)
        else:
            bins = binning
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
        if common_bins:
            # Retain the unique bin edges as an array
            bin_edges = Series(bin_edges.iloc[0])

    return ConditionalDistributionResult(
        vals=distribs,
        bins=Series(bin_edges),
        categorical=as_categorical,
        binning=binning,
        common_bins=common_bins,
    )


class ConditionalDistributionResult:
    """Result of the conditional distribution computation for a single column of data.

    Attributes
    ----------
    vals : Series
        A Series listing the bin counts for each group, indexed by (<true_label>,
        <predicted_label>, <bin_label>).
    bins: Series
        A Series listing the bin endpoints. If `common_bins` is `False`,
        this should be indexed by (<true_label>, <predicted_label>) and list
        the endpoints for each group. If the data was treated as numeric, this
        will have length `len(vals)+1` (within each group), otherwise
        `len(vals)`.
    categorical : bool
        Was the feature treated as categorical?
    binning : str
        The binning scheme used
    common_bins : bool
        Were common bins used across all groups?
    """

    def __init__(self, vals, bins, categorical, binning, common_bins):
        self.vals = vals
        self.bins = bins
        self.categorical = categorical
        self.binning = binning
        self.common_bins = common_bins

    def display_result(self, xlab):
        """Display the distributions for the given data column.

        Parameters
        ----------
        xlab : str
            Label to display on the x-axis.
        """

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

    Attributes
    ----------
    model :
        The ClassificationModel to run the evaluation for.
    test_dataset : presc.dataset.Dataset
        A Dataset to use for evaluation.
    settings : dict
        An optional dict specifying option values under `evaluations.conditional_distribution`,
        eg. `{"computation.binning": 5}`, These are restricted to the class instance and do not change the global config.
    config : presc.configuration.PrescConfig
        An optional PrescConfig instance to read options from. This will be overridden by `settings` values.
    """

    def __init__(self, model, test_dataset, settings=None, config=None):
        source_config = config or global_config
        self._config = PrescConfig(source_config)
        if settings:
            self._config.set({"evaluations": {"conditional_distribution": settings}})

        self._model = model
        self._test_dataset = test_dataset
        self._test_pred = self._model.predict_labels(test_dataset).rename("predicted")

    def compute_for_column(self, colname, **kwargs):
        """Compute the evaluation for the given dataset column.

        Parameters
        ----------
        colname : str
            A column in the dataset to compute distributions for.
        kwargs:
            On-the-fly overrides to the config option values for the computation.

        Returns
        -------
        ConditionalDistributionResult
        """
        comp_config = PrescConfig(self._config)
        comp_config = comp_config["evaluations"]["conditional_distribution"][
            "computation"
        ]
        col_overrides = comp_config["columns"][colname]
        try:
            col_overrides = col_overrides.get()
        except ConfigError:
            col_overrides = None
        if col_overrides:
            comp_config.set(col_overrides)
        if kwargs:
            comp_config.set(kwargs)

        return compute_conditional_distribution(
            data_col=self._test_dataset.df[colname],
            true_labs=self._test_dataset.labels,
            pred_labs=self._test_pred,
            as_categorical=comp_config["as_categorical"].get(bool),
            binning=comp_config["binning"].get(),
            common_bins=comp_config["common_bins"].get(bool),
        )

    def display(self, colnames=None):
        """Computes and displays the conditional distribution result for each
        specified column.

        Parameters
        ----------
        colnames : list of str
            A list of column names to run the evaluation over, creating a plot
            for each. If not supplied, defaults to columns specifed in the config.
        """
        if colnames:
            incl = colnames
            excl = None
        else:
            eval_config = self._config["evaluations"]["conditional_distribution"]
            incl = eval_config["columns_include"].get()
            excl = eval_config["columns_exclude"].get()
        cols = include_exclude_list(
            self._test_dataset.column_names, included=incl, excluded=excl
        )

        for colname in cols:
            eval_result = self.compute_for_column(colname)
            eval_result.display_result(xlab=colname)
