import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def plot_calibration_curve(est, name, X_test, y_test):
    """Plot calibration curve for est w/o and with calibration.
    inputs:
    est : fitted classifiers list
    name: classifier names"""
    fig_index = len(est)
    fig = plt.figure(fig_index, figsize=(6, 6))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in zip(est, name):
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10
        )

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="o",
            linewidth=1,
            label=name,
        )

    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlabel("Mean Predicted probability")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration plots  (reliability curve)")

    plt.tight_layout()
    plt.show()
