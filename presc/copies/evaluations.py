from sklearn.metrics import accuracy_score


def empirical_fidelity_error(y_pred_original, y_pred_copy):
    error = 1 - accuracy_score(y_pred_original, y_pred_copy)
    return error
