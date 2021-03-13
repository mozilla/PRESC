# Evaluations

Evaluations are the core functionality of the PRESC package.
Each evaluation methodology applies to a trained classification model and one or
more datasets, and produces a visual output.


## Conditional metrics

This computes standard performance scores within different partitions of a test
dataset. For example, rather than reporting an overall accuracy score for a test
dataset, the score can be computed as, say, a function over the different values
of a feature. This can show evidence of bias if the performance score differs
significantly between different areas of the feature space.

### API

The computation is maintained in `presc.evaluations.conditional_metric`, and the
main entrypoint is `ConditionalMetric`.

```python
from presc.evaluations.conditional_metric import ConditionalMetric

ecm = ConditionalMetric(cm, test_dataset)
# Plot the results for all columns in the dataset.
ecm.display()
```

For a given column in the test set, its values are binned as for a histogram,
and a scikit-learn performance score (default: accuracy) is computed for the
subset of the test dataset corresponding to each bin.

By default, the computation is run for every column in the dataset, including
all features and any other columns. For example, this can be used to create a
calibration-style plot by including a column of predicted classification
probabilities.

Columns can be selected by passing a list of column names to `ecm.display()`.

Computation for an individual column can be accessed using
`ecm.compute_for_column()`. This returns a `ConditionalMetricResult` object
which bundles the numerical results and option settings used, as well as
exposing a `display_result()` method to produce the default plot.
The underlying computation can be accessed from the `compute_conditional_metric`
module function.

### Configuration

Settings for the conditional metric evaluation are as follows:

```yaml
evaluations:
  conditional_metric:
    # Dataset columns to run the evaluation for.
    # Follows the same logic as for report evaluations.
    # "*" means 'all feature and other columns'.
    # Results will be ordered according to `columns_include`.
    columns_include: "*"
    columns_exclude: null
    # Performance metrics to compute across the dataset subsets.
    # Should be the name of a sklearn.metrics scoring function.
    metrics:
      - function: accuracy_score
        display_name: Accuracy
    computation:
      # Number of bins for partitioning a numeric column
      num_bins: 10
      # Should bin widths correspond to quantiles of a numerical column's
      # distribution (True) or be equally-spaced over its range (False)
      quantile: False
      # Should the grouping column be treated as categorical, ie. binned on its
      # unique values? Only applies if the column is numeric
      as_categorical: False
      # A dictionary of per-column overrides for the computation options.
      # Entries should have a column name as their key and settings for the
      # options above as their value.
      columns: null
```

Per-column overrides can be specified in the `columns` entry, keyed by column
name:

```yaml
    computation:
      columns:
        col1:
          num_bins: 5
        col2:
          as_categorical: True
```

Overrides can be passed to the evaluation instance as a dict, with option names
specified relative to the evaluation:

```python
ecm = ConditionalMetric(cm, test_dataset, settings={"computation.num_bins": 5})
```

## Conditional feature distributions

This computes distributions of feature values for test datapoints belonging to
each cell of the confusion matrix.

### API

The computation is maintained in `presc.evaluations.conditional_distribution`,
and the main entrypoint is `ConditionalDistribution`.

```python
from presc.evaluations.conditional_distribution import ConditionalDistribution

ecd = ConditionalDistribution(cm, test_dataset)
# Plot the results for all columns in the dataset.
ecd.display()
```

For a given column in the test set, it values are partitioned according to which
cell of the confusion matrix each row belongs to (eg. correctly classified as
class 1, class 1 misclassified as class 2, etc), and a distributional
representation is created for each cell.

By default, the computation is run for every column in the dataset, including
all features and any other columns.

Columns can be selected by passing a list of column names to `ecd.display()`.

Computation for an individual column can be accessed using
`ecd.compute_for_column()`. This returns a `ConditionalDistributionResult`
object which bundles the numerical results and option settings used, as well as
exposing a `display_result()` method to produce the default plot. The underlying
computation can be accessed from the `compute_conditional_distribution` module
function.

### Configuration

Settings for the conditional distribution evaluation are as follows:

```yaml
evaluations:
  conditional_distribution:
    # Dataset columns to run the evaluation for.
    # Follows the same logic as for report evaluations.
    # "*" means 'all feature and other columns'.
    columns_include: "*"
    columns_exclude: null
    computation:
      # Binning scheme to use for a numerical column, passed to `numpy.histogram`.
      # Can be a fixed number of bins or a string indicating a binning scheme
      binning: fd
      # Should the bins be computed over the entire column and shared across
      # groups (True) or computed within each group (False)
      common_bins: True
      # Should the data column be treated as categorical, ie. binned on its
      # unique values? Only applies if the column is numeric
      as_categorical: False
      # A dictionary of per-column overrides for the computation options.
      # Entries should have a column name as their key and settings for the
      # options above as their value.
      columns: null
```

Per-column overrides can be specified in the `columns` entry, keyed by column
name:

```yaml
    computation:
      columns:
        col1:
          binning: 10
        col2:
          as_categorical: True
```

Overrides can be passed to the evaluation instance as a dict, with option names
specified relative to the evaluation:

```python
ecd = ConditionalDistribution(cm, test_dataset, settings={"computation.binning": 5})
```

## Spatial distributions

This provides a view into the distribution of misclassified test datapoints in
feature space. For each test datapoint, it computes pairwise distances with
every training point, and an summary statistic (default: mean) of these
distances is shown, faceted according to the classes of the test points and
training points, and whether the test point was misclassified.

This helps to investigate whether misclassified points tend to be in different
areas of the feature space from correctly classified points, depending on their
class.


### API

The computation is maintained in `presc.evaluations.spatial_distribution`, and
the main entrypoint is `SpatialDistribution`.

```python
from presc.evaluations.spatial_distribution import SpatialDistribution

esd = SpatialDistribution(cm, test_dataset, train_dataset)
# Plot the results
ecm.display()
```

Computation can be accessed using `esd.compute()`. This returns a
`SpatialDistributionResult` object which bundles the numerical results and
option settings used, as well as exposing a `display_result()` method to produce
the default plot. The underlying computation can be accessed from the
`compute_spatial_distribution` module function.

### Configuration

Settings for the spatial distribution evaluation are as follows:

```yaml
evaluations:
  spatial_distribution:
    # Dataset columns to run the evaluation for.
    # Follows the same logic as for report evaluations.
    # "*" means 'all feature columns'.
    features_include: "*"
    features_exclude: null
    # The default pairwise distance metric to use for numerical features.
    distance_metric_numerical: "euclidean"
    # The default pairwise distance metric to use for categorical features.
    distance_metric_categorical: "hamming"
    # The aggregation function to use to summarize distances within each
    # class.
    summary_agg: "mean"
    # A dictionary of per-column overrides.
    # Entries should have a column name as their key and settings for the
    # options above as their value.
    columns: null
```

Distance metrics should be the name of a metric accepted by
`sklearn.metrics.pairwise_distances()`.
The aggregation summary function should be the name of a Pandas `agg` function.
Currently, only Hamming distance (ie. 0-1 dissimilarity) is accepted for
categorical features.

Per-column overrides can be specified in the `columns` entry, keyed by column
name:

```yaml
  columns:
    col1:
      distance_metric_numerical: "l1"
```

Overrides can be passed to the evaluation instance as a dict, with option names
specified relative to the evaluation:

```python
esd = SpatialDistribution(cm, test_dataset, train_dataset,
settings={"summary_agg": "median"})
```

## Train-test splits

This offers a view into the degree to which the classifier performance is
affected by the choice of train-test split proportion.
This is accomplished by simulating different splits and computing a performance
score as a function of split proportion.

It should be emphasized that the goal is not to select the proportion which
maximizes the performance score, but rather to check how much the results would
have been impacted if a different split were used.


### API

The computation is maintained in `presc.evaluations.train_test_splits`, and
the main entrypoint is `TrainTestSplits`.

```python
from presc.evaluations.train_test_splits import TrainTestSplits

etts = TrainTestSplits(cm, train_dataset)
# Plot the results
ecm.display()
```

The given training set is split into train and test parts using a split
proportion varying at regular intervals. The model is retrained on the training
part and scored on the test part, with this result replicated multiple times
(similar to repeated cross-validation). A summary of these score distributions
is then presented visually.

Computation can be accessed using `esd.compute()`. This returns a
`TrainTestSplitsResult` object which bundles the numerical results and
option settings used, as well as exposing a `display_result()` method to produce
the default plot. The underlying computation can be accessed from the
`compute_train_test_splits()` module function.

### Configuration

Settings for the train-test splits evaluation are as follows:

```yaml
evaluations:
  train_test_splits:
    # Scoring function used to evaluate test performance.
    # Should be a string recognized by `sklearn.model_selection.cross_val_score`
    metrics:
      - function: accuracy
        display_name: Accuracy
    computation:
      # Increment between train-test split ratios
      split_size_increment: 0.05
      # Number of random replicates to run for each split
      num_replicates: 20
      # Set the random state for reproducibility
      random_state: 543
```

Scoring metrics should be the name of a metric defined in `sklearn.metrics`.

Overrides can be passed to the evaluation instance as a dict, with option names
specified relative to the evaluation:

```python
etts = TrainTestSplits(cm, train_dataset, settings={"num_replicates": 10})
```

