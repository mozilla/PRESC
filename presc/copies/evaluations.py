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
