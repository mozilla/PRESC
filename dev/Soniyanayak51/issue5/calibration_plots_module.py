import seaborn as sns
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


def calibration_plot_for_single_classifier(clfModel, clfName, y_test, X_test):
    """
    Plots a calibration curve for a single classifier
    Args: clfModel-Classification model, clfName-The classifier's name, y_test-Test labels, X_test-Test data
    """
    y_test_predict_proba = clfModel.predict_proba(X_test)[
        :, 1
    ]  # The calibration_curve implementation expects just one of the classes in an array.

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_test_predict_proba, n_bins=10
    )
    fig, ax = plt.subplots(1, figsize=(12, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "--", color="red")  # Perfectly calibrated

    # sns.despine(left=True, bottom=True)
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")
    ax.set_xlabel("Predicted prob")
    ax.set_ylabel("True probability in each bin")
    plt.title(clfName + " Calibration Curve", fontsize=20)


def calibration_plot_for_multiple_classifier(
    classifierList, classifierNameList, y_test, X_test
):
    """
    Plots a calibration curve for multiple classifiers.
    Args: classifierList-Classification model list, classifierNameList-The classifiers names, y_test-Test labels, X_test-Test data
    """
    i = 0
    fraction_positives = list()
    mean_predicted_values = list()
    fig, ax = plt.subplots()
    for clfModel in classifierList:
        y_test_predict_proba = clfModel.predict_proba(X_test)[:, 1]
        fraction_of_positive, mean_predicted_value = calibration_curve(
            y_test, y_test_predict_proba, n_bins=10
        )
        plt.plot(
            mean_predicted_value,
            fraction_of_positive,
            marker="o",
            label=classifierNameList[i],
        )
        i = i + 1
    plt.plot([0, 1], [0, 1], "--", color="red")  # Perfectly calibrated
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")
    ax.set_xlabel("Predicted prob")
    ax.set_ylabel("True probability in each bin")
    plt.title(" Calibration Curve for multiple classifers", fontsize=15)
    plt.legend()
    plt.show()
