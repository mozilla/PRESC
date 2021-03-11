from presc.evaluations.utils import get_bins, is_discrete
from presc.utils import include_exclude_list
from presc.configuration import PrescConfig
from presc import global_config

from pandas import DataFrame, Series
import sklearn.metrics as sk
import matplotlib.pyplot as plt
from confuse import ConfigError, NotFoundError, ConfigTypeError


def compute_conditional_metric(
    grouping_col,
    true_labs,
    pred_labs,
    metric,
    as_categorical=False,
    num_bins=10,
    quantile=False,
):
    """Compute metric values conditional on the grouping column.

    The metric is computed within unique values of the grouping column
    (categorical) or within bins partitioning its range (continuous).

    Parameters
    ----------
    grouping_col : Series
        Series defining a grouping for the metric computation.
    true_labs : Series
        Series of true labels for a test dataset.
    pred_labs : Series
        Series of labels predicted by a model for a test dataset.
    metric : function
        The evaluation metric to compute across the groupings. This should be
        a function f(y_true, y_pred) which accepts Series of true and
        predicted labels.
    as_categorical : bool
        Should the grouping column be treated as categorical, ie. binned
        on its unique values? If it is not numeric, this param is ignored.
    num_bins : int
        Number of bins to use for grouping a numeric column.
    quantile : bool
        Should the bin widths correspond to quantiles of a numerical column's
        distribution (`True`) or be equally-spaced over its range (`False`).

    Returns
    -------
    ConditionalMetricResult
    """

    y_vals = DataFrame({"y_true": true_labs, "y_pred": pred_labs})
    if is_discrete(grouping_col):
        as_categorical = True
    if as_categorical:
        grouping = grouping_col
        bins = grouping.unique()
    else:
        grouping, bins = get_bins(grouping_col, num_bins, quantile)
    binned_metric_vals = y_vals.groupby(grouping).apply(
        lambda gp: metric(gp["y_true"], gp["y_pred"])
    )

    return ConditionalMetricResult(
        vals=binned_metric_vals,
        bins=Series(bins),
        categorical=as_categorical,
        num_bins=num_bins,
        quantile=quantile,
    )


def _get_metrics_for_column(colname, eval_config):
    default_metrics = eval_config["metrics"].get()
    metrics_to_use = default_metrics
    try:
        col_metrics = eval_config["computation"]["columns"][colname]["metrics"].get()
        metrics_to_use = col_metrics
    except NotFoundError:
        pass
    except ConfigTypeError:
        pass

    metrics = []
    for metric_to_use in metrics_to_use:
        function_name = metric_to_use.get("function")
        display_name = metric_to_use.get("display_name", function_name)
        try:
            # TODO expand to non sklearn functions.
            metric_function = getattr(sk, function_name)
            metrics.append({"function": metric_function, "display_name": display_name})
        except AttributeError:
            print(
                f"Column: `{colname}` Function: `{function_name}` is not a valid sklearn metric. "
                f"\nVerify evaluations.conditional_metric.metrics configuration and/or "
                f"\nevaluations.conditional_metric.computation.columns.{colname} (if provided)."
            )
    return metrics


class ConditionalMetricResult:
    """Result of the conditional metric evaluation for a single grouping.

    Attributes
    ----------
    vals : Series
        A Series listing the computation result for each bin.
    bins: Series
        A Series listing the bin endpoints. If the feature was treated as
        numeric, this will have length `len(vals)+1`, otherwise `len(vals)`.
    categorical : bool
        Aas the feature treated as categorical?
    num_bins : int
        Number of bins used for grouping.
    quantile: bool
        Was grouping quantile-based?
    """

    def __init__(self, vals, bins, categorical, num_bins, quantile):
        self.vals = vals
        self.bins = bins
        self.categorical = categorical
        self.num_bins = num_bins
        self.quantile = quantile

    def display_result(self, xlab, ylab):
        """Display the evaluation result for the given grouping and metric.

        Parameters
        ----------
        xlab : str
            Label to display on the x-axis.
        ylab: str
            Label to display on the y-axis.

        """

        if self.categorical:
            result_edges = self.bins.astype("str")
            alignment = "center"
            widths = 1
        else:
            result_edges = self.bins[:-1]
            alignment = "edge"
            # First element will be NaN.
            widths = self.bins.diff()[1:]

        plt.ylim(0, 1)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.bar(
            result_edges,
            self.vals,
            width=widths,
            bottom=None,
            align=alignment,
            edgecolor="white",
            linewidth=2,
        )
        plt.show(block=False)


class ConditionalMetric:
    """Computation of confusion-based metrics across subsets of a test dataset.

    Attributes
    ----------
    model:
        The ClassificationModel to run the evaluation for.
    test_dataset :  presc.dataset.Dataset
        A Dataset to use for evaluation.
    settings: dict
        An optional dict specifying option values under
        `evaluations.conditional_metric`, eg. `{"computation.num_bins": 5}`
        These are restricted to the class instance and do not change the global config.
    config: presc.configuration.PrescConfig
        An optional PrescConfig instance to read options from. This will be
        overridden by `settings` values.
    """

    def __init__(self, model, test_dataset, settings=None, config=None):
        source_config = config or global_config
        self._config = PrescConfig(source_config)
        if settings:
            self._config.set({"evaluations": {"conditional_metric": settings}})

        self._model = model
        self._test_dataset = test_dataset
        self._test_pred = self._model.predict_labels(test_dataset)

    def compute_for_column(self, colname, metric, **kwargs):
        """Compute the evaluation for the given dataset column.

        The metric is computed within unique values of the specified column
        (if categorical) or within bins partitioning its range (if continuous).

        colname : str
            A column in the dataset to partition on.
        metric : function
            The evaluation metric to compute across the partitions. This should be
            a function f(y_true, y_pred) which accepts Series of true and
            predicted labels.
        kwargs :
            On-the-fly overrides to the config option values for the computation.

        Returns
        ------
        ConditionalMetricResult
        """
        comp_config = PrescConfig(self._config)
        comp_config = comp_config["evaluations"]["conditional_metric"]["computation"]
        col_overrides = comp_config["columns"][colname]
        try:
            col_overrides = col_overrides.get()
        except ConfigError:
            col_overrides = None
        if col_overrides:
            comp_config.set(col_overrides)
        if kwargs:
            comp_config.set(kwargs)

        return compute_conditional_metric(
            grouping_col=self._test_dataset.df[colname],
            true_labs=self._test_dataset.labels,
            pred_labs=self._test_pred,
            metric=metric,
            as_categorical=comp_config["as_categorical"].get(bool),
            num_bins=comp_config["num_bins"].get(int),
            quantile=comp_config["quantile"].get(bool),
        )

    def display(self, colnames=None):
        """Computes and displays the conditional metric result for each specified column.

        Parameters
        ----------
        colnames : list of str
            A list of column names to run the evaluation over, creating a plot
            for each. If not supplied, defaults to columns specifed in the config.
        metric_name : str
            Display name identifying the metric to show on the plot
        """
        eval_config = self._config["evaluations"]["conditional_metric"]
        if colnames:
            incl = colnames
            excl = None
        else:
            incl = eval_config["columns_include"].get()
            excl = eval_config["columns_exclude"].get()
        cols = include_exclude_list(
            self._test_dataset.column_names, included=incl, excluded=excl
        )

        for colname in cols:
            metrics = _get_metrics_for_column(colname=colname, eval_config=eval_config)
            for metric in metrics:
                function = metric.get("function")
                display_name = metric.get("display_name")
                eval_result = self.compute_for_column(colname, metric=function)
                eval_result.display_result(xlab=colname, ylab=display_name)
