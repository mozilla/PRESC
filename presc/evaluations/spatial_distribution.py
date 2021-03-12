from collections import defaultdict

from pandas import DataFrame, Series, MultiIndex
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns
from confuse import ConfigError

from presc.evaluations.utils import is_discrete
from presc.utils import include_exclude_list
from presc.configuration import PrescConfig
from presc import global_config


def _pairwise_dist(df1, df2, metrics):
    """Compute pairwise distances between the rows of two DFs with common columns.

    Parameters
    ----------
    df1 : DataFrame
        A DF with dimensions n x p.
    df2 : DataFrame
        A DF with dimensions m x p.
    metrics : dict
        Mapping of column names to strings identifying the distance metric to
        use for entries in that column. Strings must be valid values accepted by
        `sklearn.metrics.pairwise_distances`.

    Returns
    -------
    numpy.array
        Array with dimensions n x m listing the distance between df1[i] and
        df2[j]. If different metrics are used for different subsets of columns,
        they are combined by adding.
    """
    cols_for_metric = defaultdict(list)
    for c, m in metrics.items():
        cols_for_metric[m].append(c)

    pairwise_dist = None
    for m, cols in cols_for_metric.items():
        pwd = pairwise_distances(df1[cols], df2[cols], metric=m)
        if pairwise_dist is None:
            pairwise_dist = pwd
        else:
            pairwise_dist += pwd

    return pairwise_dist


def compute_spatial_distribution(
    test_features,
    test_labs_true,
    test_labs_pred,
    base_features,
    base_labs,
    numerical_dist_metric=None,
    categorical_dist_metric=None,
    summary="mean",
):
    """Compute a summary of the pairwise distances between points.

    This computes pairwise distances between the test points and base points in
    feature space (ie. how far each test point is from each base point), and
    returns a summary of the distance for each test point relative to each base
    class.

    Parameters
    ----------
    test_features : DataFrame
        Feature values for the test dataset.
    test_labs_true : Series
        True labels for the test dataset.
    test_labs_pred : Series
        Labels predicted by a model for the test dataset.
    base_features:  DataFrame
        Feature values for the base dataset.
    base_labs : Series
        True labels for the base dataset.
    numerical_dist_metric : dict
        The metrics to use to measure distance between numerical
        (continuous)-valued columns. This should be a dict mapping column names
        to strings, each a named metric as accepted by
        `sklearn.metrics.pairwise_distances` appropriate for continuous data
    categorical_dist_metric : dict
        The metrics to use to measure distance between categorical
        (discrete)-valued columns. This should be a dict mapping column names to
        strings, each a named metric as accepted by
        `sklearn.metrics.pairwise_distances` appropriate for discrete data
    summary : str
        An aggregation function to apply to a Pandas Grouped object.

    Only columns listed in the distance metric dists will be included in the
    distance computation.

    Returns
    -------
    SpatialDistributionResult
    """
    # Compute a DF of pairwise distances between base datapoints (rows)
    # and test datapoints (cols).
    pairwise_dist = None
    if numerical_dist_metric:
        # Normalize numeric features to reduce the effect of different scales
        num_cols = list(numerical_dist_metric.keys())
        scaler = StandardScaler()
        base_scaled = DataFrame(
            scaler.fit_transform(base_features[num_cols]), columns=num_cols
        )
        test_scaled = DataFrame(
            scaler.transform(test_features[num_cols]), columns=num_cols
        )
        pairwise_dist = _pairwise_dist(base_scaled, test_scaled, numerical_dist_metric)
    if categorical_dist_metric:
        categ_cols = list(categorical_dist_metric.keys())
        encoder = OrdinalEncoder()
        base_encoded = DataFrame(
            encoder.fit_transform(base_features[categ_cols]), columns=categ_cols
        )
        test_encoded = DataFrame(
            encoder.transform(test_features[categ_cols]), columns=categ_cols
        )
        pairwise_dist_categ = _pairwise_dist(
            base_encoded, test_encoded, categorical_dist_metric
        )
        if pairwise_dist is None:
            pairwise_dist = pairwise_dist_categ
        else:
            pairwise_dist += pairwise_dist_categ
    df_dist = DataFrame(
        pairwise_dist,
        index=base_labs.index,
        columns=test_labs_true.index,
    )

    # Summarize distances within each base dataset class separately for each
    # test datapoint.
    # Result is a m x k DF with 1 row for each test datapoint and 1 column for
    # each base class.
    df_summ = df_dist.groupby(base_labs).agg(summary).transpose()
    # Add the test labels to the index for easy reference.
    df_summ = df_summ.set_index(
        MultiIndex.from_arrays([test_labs_true, test_labs_pred, df_summ.index])
    )

    return SpatialDistributionResult(
        vals=df_summ,
        dist_metrics_num=numerical_dist_metric,
        dist_metrics_categ=categorical_dist_metric,
        summary=summary,
    )


class SpatialDistributionResult:
    """Result of the spatial distribution computation.

    Attributes
    ----------
    vals : DataFrame
        A  DataFrame listing the summary values for each test datapoint, indexed
        by (<true_label>, <predicted_label>, <datapoint_index>).
    dist_metrics_num : dict
        Mapping of numerical column names to the metric that was used.
    dist_metrics_categ : dict
        Mapping of categorical column names to the metric that was used.
    summary : str
        The summary used to aggregate distances for each class
    """

    def __init__(self, vals, dist_metrics_num, dist_metrics_categ, summary):
        self.vals = vals
        self.dist_metrics_num = dist_metrics_num
        self.dist_metrics_categ = dist_metrics_categ
        self.summary = summary

    def display_result(self):
        """Display the distances summaries as scatterplots."""
        ind = self.vals.index
        is_misclassified = Series(
            ind.get_level_values(0) != ind.get_level_values(1), index=ind
        )
        sns.set()
        for y_true in self.vals.index.get_level_values(0).unique():
            df_class = self.vals.loc[y_true]
            df_dist = df_class.drop(columns=[y_true]).stack().rename("dist_other")
            df_dist.index = df_dist.index.set_names("base_class", level=-1)
            df_dist = df_dist.reset_index(level="base_class")
            df_dist["dist_same"] = df_class[y_true]
            df_dist["is_misclassified"] = is_misclassified.loc[y_true]

            (
                sns.relplot(
                    data=df_dist,
                    x="dist_same",
                    y="dist_other",
                    col="base_class",
                    hue="is_misclassified",
                    height=7,
                )
                .set_titles(
                    f"Distance to base dataset classes (other = {{col_name}})\nfor test datapoints (class = {y_true})"
                )
                .set_xlabels(f"Distance to base class {y_true}")
                .set_ylabels("Distance to other base class")
            )


def _get_distance_metrics_by_column(features_num, features_categ, eval_config):
    """Determine the distance metric to use for each dataset feature.

    Parameters
    ----------
    features_num : list
        Numerical feature names to include in the distance computation.
    features_categ : list
        Categorical feature names to include in the distance computation.
    eval_config : presc.configuration.PrescConfig
        The config subview corresponding to this evaluation.

    Returns
    -------
    dict, dict
        Two dicts mapping feature names to metrics, either a column-specifc
        metric, if any, or the default metric for the feature type. The first
        lists metrics for numerical features and the second for categorical
        features. Only features listed in the input lists will be included.
    """
    def_dist_num = eval_config["distance_metric_numerical"].get()
    def_dist_categ = eval_config["distance_metric_categorical"].get()
    dist_metrics_num = {}
    dist_metrics_categ = {}
    included_features = features_num + features_categ
    try:
        col_overrides = eval_config["columns"].get()
    except ConfigError:
        col_overrides = None

    if col_overrides:
        for col, settings in col_overrides.items():
            if col not in included_features:
                continue
            if "distance_metric_categorical" in settings:
                dist_metrics_categ[col] = settings["distance_metric_categorical"]
            elif "distance_metric_numerical" in settings:
                dist_metrics_num[col] = settings["distance_metric_numerical"]
            elif settings.get("as_categorical", False):
                dist_metrics_categ[col] = def_dist_categ

    overridden_features = list(dist_metrics_num.keys()) + list(
        dist_metrics_categ.keys()
    )
    for col in features_num:
        if col not in overridden_features:
            dist_metrics_num[col] = def_dist_num
    for col in features_categ:
        if col not in overridden_features:
            dist_metrics_categ[col] = def_dist_categ

    return dist_metrics_num, dist_metrics_categ


class SpatialDistribution:
    """Computation of distributions of data in feature space.

    Attributes
    ----------
    model: presc.model.ClassificationModel
        The ClassificationModel to run the evaluation for.
    test_dataset :  presc.dataset.Dataset
        A Dataset to use for evaluation.
    train_dataset :  presc.dataset.Dataset
        A Dataset to use as the baseline for distance measures (eg. the training
        data).
    settings: dict
        An optional dict specifying option values under
        `evaluations.spatial_distribution`, eg. `{"summary_agg": "median"}`
        These are restricted to the class instance and do not change the global
        config.
    config: presc.configuration.PrescConfig
        An optional PrescConfig instance to read options from. This will be
        overridden by `settings` values.
    """

    def __init__(self, model, test_dataset, train_dataset, settings=None, config=None):
        source_config = config or global_config
        self._config = PrescConfig(source_config)
        if settings:
            self._config.set({"evaluations": {"spatial_distribution": settings}})

        self._model = model
        self._test_dataset = test_dataset
        self._test_pred = self._model.predict_labels(test_dataset).rename("predicted")
        self._train_dataset = train_dataset

    def compute(self, **kwargs):
        """Compute the evaluation for the given datasets.

        Parameters
        ----------
        kwargs:
            On-the-fly overrides to the config option values for the computation.

        Returns
        -------
        SpatialDistributionResult
        """
        eval_config = PrescConfig(self._config)
        eval_config = eval_config["evaluations"]["spatial_distribution"]
        if kwargs:
            eval_config.set(kwargs)

        # Feature columns to include in the distance computation.
        feats_incl = eval_config["features_include"].get()
        feats_excl = eval_config["features_exclude"].get()
        feats = include_exclude_list(
            self._test_dataset.feature_names, included=feats_incl, excluded=feats_excl
        )
        num_feats = []
        categ_feats = []
        for col in feats:
            if is_discrete(self._test_dataset.features[col]):
                categ_feats.append(col)
            else:
                num_feats.append(col)

        # Figure the metric to use for each feature.
        dist_metrics_num, dist_metrics_categ = _get_distance_metrics_by_column(
            num_feats, categ_feats, eval_config
        )

        return compute_spatial_distribution(
            test_features=self._test_dataset.features,
            test_labs_true=self._test_dataset.labels,
            test_labs_pred=self._test_pred,
            base_features=self._train_dataset.features,
            base_labs=self._train_dataset.labels,
            numerical_dist_metric=dist_metrics_num,
            categorical_dist_metric=dist_metrics_categ,
            summary=eval_config["summary_agg"].get(),
        )

    def display(self):
        """Computes and displays the spatial distribution results."""
        eval_result = self.compute()
        eval_result.display_result()
