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
