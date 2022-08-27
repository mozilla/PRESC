import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

from presc.dataset import Dataset


def empirical_fidelity_error(y_pred_original, y_pred_copy):
    """Computes the empirical fidelity error of a classifier copy.

    Quantifies the resemblance of the copy to the original classifier. This
    value is zero when the copy makes exactly the same predictions than the
    original classifier (including misclassifications).

    Parameters
    ----------
    y_pred_original : list or 1d array-like
        Predicted labels, as returned by the original classifier.
    y_pred_copy : list or 1d array-like
        Predicted labels, as returned by the classifier copy.

    Returns
    -------
    float
        The numerical value of the empirical fidelity error.
    """
    error = 1 - accuracy_score(y_pred_original, y_pred_copy)
    return error


def replacement_capability(y_true, y_pred_original, y_pred_copy):
    """Computes the replacement capability of a classifier copy.

    Quantifies the ability of the copy model to substitute the original model,
    i.e. maintaining the same accuracy in its predictions. This value is one
    when the accuracy of the copy model is the same as the original model,
    although the individual predictions may be different, approaching zero if
    the accuracy of the copy is much smaller than the original, and it can even
    take values larger than one if the copy model is better than the original.

    Parameters
    ----------
    y_true : list or 1d array-like
        True labels from the data.
    y_pred_original : list or 1d array-like
        Predicted labels, as returned by the original classifier.
    y_pred_copy : list or 1d array-like
        Predicted labels, as returned by the classifier copy.

    Returns
    -------
    float
        The numerical value of the replacement capability.
    """
    accuracy_original = accuracy_score(y_true, y_pred_original)
    accuracy_copy = accuracy_score(y_true, y_pred_copy)
    rcapability = accuracy_copy / accuracy_original
    return rcapability


def summary_metrics(
    original_model=None,
    copy_model=None,
    test_data=None,
    synthetic_data=None,
    show_results=True,
):
    """Computes several metrics to evaluate the classifier copy.

    Summary of metrics that evaluate the quality of a classifier copy, not only
    to assess its performance as classifier but to quantify its resemblance to
    the original classifier. Accuracy of the original and the copy models (using
    the original test data), and the empirical fidelity error and replacement
    capability of the copy (using the original test data and/or the generated
    synthetic data).

    Parameters
    ----------
    original_model : sklearn-type classifier
        Original ML classifier to be copied.
    copy_model : presc.copies.copying.ClassifierCopy
        ML classifier copy from the original ML classifier.
    test_data : presc.dataset.Dataset
        Subset of the original data reserved for testing.
    synthetic_data : presc.dataset.Dataset
        Synthetic data generated using the original model.
    show_results : bool
        If `True` the metrics are also printed.

    Returns
    -------
    dict
        The values of all metrics.
    """

    results = {
        "Original Model Accuracy (test)": None,
        "Copy Model Accuracy (test)": None,
        "Empirical Fidelity Error (synthetic)": None,
        "Empirical Fidelity Error (test)": None,
        "Replacement Capability (synthetic)": None,
        "Replacement Capability (test)": None,
    }

    if test_data is not None:
        if original_model is not None:
            y_pred_original_test = original_model.predict(test_data.features)

            original_accuracy = accuracy_score(test_data.labels, y_pred_original_test)
            results["Original Model Accuracy (test)"] = original_accuracy

        if copy_model is not None:
            y_pred_copy_test = copy_model.copy.predict(test_data.features)

            copy_accuracy = accuracy_score(test_data.labels, y_pred_copy_test)
            results["Copy Model Accuracy (test)"] = copy_accuracy

            efe_test = copy_model.compute_fidelity_error(test_data=test_data.features)
            results["Empirical Fidelity Error (test)"] = efe_test

        if (original_model is not None) and (copy_model is not None):
            rc_test = replacement_capability(
                test_data.labels, y_pred_original_test, y_pred_copy_test
            )
            results["Replacement Capability (test)"] = rc_test

    if synthetic_data is not None:
        if original_model is not None:
            y_pred_original_synth = original_model.predict(synthetic_data.features)

        if copy_model is not None:
            y_pred_copy_synth = copy_model.copy.predict(synthetic_data.features)

            efe_synthetic = copy_model.compute_fidelity_error(
                test_data=synthetic_data.features
            )
            results["Empirical Fidelity Error (synthetic)"] = efe_synthetic

        if (original_model is not None) and (copy_model is not None):
            rc_synthetic = replacement_capability(
                synthetic_data.labels, y_pred_original_synth, y_pred_copy_synth
            )
            results["Replacement Capability (synthetic)"] = rc_synthetic

    if show_results:
        for name, value in results.items():
            if value is not None:
                print(f"{name:<37}   {value:.4f}")

    return results


def multivariable_density_comparison(
    datasets=[None],
    feature1=None,
    feature2=None,
    label_col="class",
    titles=None,
    other_kwargs={
        "alpha": 0.3,
        "common_norm": False,
        "fill": True,
        "n_levels": 4,
        "legend": False,
    },
):
    """Visualization to compare class density projections in detail.

    Allows to compare the different topologies of a number of ML classifiers in
    a multivariable feature space by choosing a feature pair and "squashing" the
    rest of the features into a projected density distribution for each class.

    It is important that the classifier datasets are obtained through a
    homogeneous sampling throughout the feature space to avoid introducing
    spurious shapes in the projected density distributions. `uniform_sampling`
    is a good option for that.

    `normal_sampling` and any other non-uniform samplers should be avoided
    because the intrinsic class distributions become convoluted with its
    gaussian shape obscuring them. Note that `grid_sampling` is also not
    recommended because it samples very specific interval points and thus yields
    density peaks.

    Parameters
    ----------
    datasets : list of pandas DataFrames
        List of the datasets with the sampled and labeled points for each
        classifier included in the comparison.
    feature1 :
        Name of feature to display in the x-axis.
    feature2 :
        Name of feature to display in the y-axis.
    label_col : str
        Name of the label column.
    titles : list of str
        List of names to identify each classifier and label their subplot.
    **other_kwargs : dict
        Any other seaborn.kdeplot parameters needed to adjust the visualization.
        Default parameters are {"alpha": 0.3, "common_norm": False, "fill": True,
        "n_levels": 4, "legend": False}. The value of any parameter specified
        within the other_kwargs dictionary will be overwritten, including any
        of these.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the detailed classifier comparison.
    matplotlib.axes.Axes or array of Axes
        Contains most of the figure elements of the classifier comparison and
        sets the coordinate system.
    """
    kdeplot_kwargs = {
        "alpha": 0.3,
        "common_norm": False,
        "fill": True,
        "n_levels": 4,
        "legend": False,
    }
    for key, value in other_kwargs.items():
        kdeplot_kwargs[key] = value

    num_comparisons = len(datasets)
    class_names = set()
    for index_models in range(num_comparisons):
        class_names = class_names | set(datasets[index_models][label_col].unique())
    class_names = sorted(list(class_names))
    max_num_classes = len(class_names)

    fig, axs = plt.subplots(
        max_num_classes,
        num_comparisons,
        figsize=(3.5 * num_comparisons, 3.5 * max_num_classes),
        sharex=True,
        sharey=True,
    )
    for index_models in range(num_comparisons):
        for index_classes, class_name in enumerate(class_names):
            axs[index_classes, index_models] = sns.kdeplot(
                x=datasets[index_models][
                    datasets[index_models][label_col] == class_name
                ][feature1],
                y=datasets[index_models][
                    datasets[index_models][label_col] == class_name
                ][feature2],
                hue=datasets[index_models][label_col],
                ax=axs[index_classes, index_models],
                **kdeplot_kwargs,
            )
            axs[index_classes, 0].set_ylabel(
                label_col + " = " + str(class_name) + "\n\n" + feature2
            )
        if titles is not None:
            axs[0, index_models].set_title(titles[index_models])
    plt.show(block=False)

    return fig, axs


def keep_top_classes(dataset, min_num_samples=2, classes_to_keep=None):
    """Function to remove rows from minoritary classes from PRESC Datasets.

    Only classes that have more than the specified minimum number of samples
    will be kept. If a list of the classes of interest is indicated, then this
    requirement is overrided.

    Parameters
    ----------
    dataset : presc.dataset.Dataset
        PRESC dataset from which we want to remove the minoritary classes.
    min_num_samples : int
        Minimum number of samples that the classes should have in order to keep
        them.
    classes_to_keep : list
        Name of the classes to keep. If a list of classes is specified here,
        then the parameter `min_num_samples` is overriden, and the specified
        classes will have any number of samples.

    Returns
    -------
    presc.dataset.Dataset
        PRESC Dataset without the samples from the minoritary classes.
    """
    label_col = dataset.labels.name
    if classes_to_keep is None:
        classes_to_keep = (
            dataset.df[label_col]
            .value_counts()[dataset.df[label_col].value_counts() >= min_num_samples]
            .index.to_list()
        )
    new_dataframe = dataset.df[dataset.df[label_col].isin(classes_to_keep)].copy()
    if new_dataframe[label_col].dtype.name == "categorical":
        print(str(label_col), " is categorical")
        new_dataframe[label_col] = new_dataframe.loc[:, label_col].cat.set_categories(
            classes_to_keep
        )

    return Dataset(new_dataframe, label_col=label_col)
