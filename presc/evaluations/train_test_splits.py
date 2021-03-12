from presc.configuration import PrescConfig
from presc import global_config

from numpy import arange
from sklearn.model_selection import ShuffleSplit, cross_val_score
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
from matplotlib import rc as mpl_rc


def compute_train_test_splits(
    dataset,
    classifier,
    metric,
    split_size_increment=0.1,
    num_replicates=10,
    random_state=543,
):
    """Evaluate classifier performance across different train-test split ratios.

    This traverses a grid of train-test splits, retraining the model on each
    training subset and computing a performance metric on the corresponding test
    set.

    Parameters
    ----------
    dataset : presc.dataset.Dataset
        The Dataset to use for the evaluation.
    classifier : sklearn Classifier
        The classifier to evaluate.
    metric : str
        The evaluation metric to compute for each split. This should be the name
        of a `sklearn.metrics` scorer function.
    split_size_increment : float
        Increment between linearly-space train-test split ratios.
    num_replicates : int
        Number of randomized replicates to run for each split.
    random_state : int
        Set the random state for reproducibility.

    Returns
    -------
    TrainTestSplitsResult
    """
    test_set_sizes = arange(
        split_size_increment, 1 - split_size_increment / 2, split_size_increment
    )

    results = []
    for test_prop in test_set_sizes:
        splitter = ShuffleSplit(
            n_splits=num_replicates, test_size=test_prop, random_state=random_state
        )
        cv_result = cross_val_score(
            classifier,
            X=dataset.features,
            y=dataset.labels,
            scoring=metric,
            cv=splitter,
        )
        results.append(cv_result)

    df = DataFrame(
        results,
        index=Series(test_set_sizes, name="test_set_size"),
        columns=[f"rep_{i}" for i in range(len(results[0]))],
    )

    return TrainTestSplitsResult(
        vals=df,
        metric=metric,
        split_size_increment=split_size_increment,
        num_replicates=num_replicates,
        random_state=random_state,
    )


class TrainTestSplitsResult:
    """Result of the train-test splits evaluation.

    Attributes
    ----------
    vals : DataFrame
        A DataFrame listing the performance score results for each split size.
        Rows are indexed by split size, with replicated results across columns.
    metric : str
        The evaluation metric used to score performance.
    split_size_increment : float
        Increment between linearly-space train-test split ratios.
    num_replicates : int
        Number of randomized replicates for each split.
    random_state : int
        Random state setting.
    """

    def __init__(
        self, vals, metric, split_size_increment, num_replicates, random_state
    ):
        self.vals = vals
        self.metric = (metric,)
        self.split_size_increment = split_size_increment
        self.num_replicates = num_replicates
        self.random_state = random_state

    def display_result(self, metric_name):
        """Display the evaluation results across split sizes.

        Parameters
        ----------
        metric_name : str
            Display name for the scoring metric.
        """

        averages = self.vals.mean(axis="columns")
        standard_deviations = self.vals.std(axis="columns")
        x = self.vals.index
        x_name = "Test set fraction"
        figure, axs = plt.subplots(1, 2, figsize=(15, 6))

        mpl_rc("font", size=12)
        mpl_rc("axes", titlesize=12)

        axs[0].plot(x, averages, color="slateblue", linewidth=2.0)
        axs[0].fill_between(
            x,
            averages - standard_deviations,
            averages + standard_deviations,
            color="lavender",
        )
        axs[0].set_xlabel(x_name)
        axs[0].set_ylabel("Average " + metric_name)

        axs[1].scatter(x, standard_deviations)
        axs[1].set_xlabel(x_name)
        axs[1].set_ylabel(metric_name.capitalize() + " standard deviation")

        plt.show(block=False)

        min_sd_ind = standard_deviations.argmin()
        print(
            f"\nIndex of point where {metric_name} has the smallest standard deviation:"
            + f" {min_sd_ind}"
            + f"\n{x_name} where {metric_name} has smallest standard deviation:"
            + f" {x[min_sd_ind]:.4f}"
            + f"\nAverage {metric_name} at {x_name} with the smallest standard deviation:"
            + f" {averages.iloc[min_sd_ind]:.4f}"
        )


class TrainTestSplits:
    """Simulation of performance across different train-test split ratios.

    Attributes
    ----------
    model :
        The ClassificationModel to run the evaluation for.
    train_dataset : presc.dataset.Dataset
        A Dataset to use for evaluation.
    settings : dict
        An optional dict specifying option values under `evaluations.conditional_distribution`,
        eg. `{"computation.num_replicates": 5}`, These are restricted to the class instance and do not change the global config.
    config : presc.configuration.PrescConfig
        An optional PrescConfig instance to read options from. This will be overridden by `settings` values.
    """

    def __init__(self, model, train_dataset, settings=None, config=None):
        source_config = config or global_config
        self._config = PrescConfig(source_config)
        if settings:
            self._config.set({"evaluations": {"train_test_splits": settings}})

        self._model = model
        self._train_dataset = train_dataset

    def compute(self, metric, **kwargs):
        """Compute the evaluation for the given dataset column.

        Parameters
        ----------
        metric : str
            The evaluation metric to compute for each split. This should be the
            name of a `sklearn.metrics` scorer function.
        kwargs:
            On-the-fly overrides to the config option values for the computation.

        Returns
        -------
        TrainTestSplitsResult
        """
        eval_config = PrescConfig(self._config)
        eval_config = eval_config["evaluations"]["train_test_splits"]["computation"]
        if kwargs:
            eval_config.set(kwargs)

        return compute_train_test_splits(
            dataset=self._train_dataset,
            classifier=self._model.classifier,
            metric=metric,
            split_size_increment=eval_config["split_size_increment"].get(float),
            num_replicates=eval_config["num_replicates"].get(int),
            random_state=eval_config["random_state"].get(int),
        )

    def display(self):
        """Computes and displays the train-test splits result for each
        specified metric."""
        for metric in self._config["evaluations"]["train_test_splits"]["metrics"].get():
            metric_func = metric["function"]
            eval_result = self.compute(metric_func)
            eval_result.display_result(
                metric_name=metric.get("display_name", metric_func)
            )
