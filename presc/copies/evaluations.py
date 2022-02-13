from sklearn.metrics import accuracy_score


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
        Predicted labels, as returned by the classifier copy.

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
            print(f"{name:<37}   {value:.4f}")

    return results
