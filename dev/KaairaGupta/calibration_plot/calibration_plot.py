from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


def plot_calibration_curves(
    classifier_model_list, classifier_model_names, features_test, labels_test, n_bins
):
    """
        Input:
            classifier_model_list: List of objects of binary classifier models which have method 
                classifier.predict_proba(features_test) with output y = list of predicted prbablities 
                for two classes (binary classification) <np.array<np.array[2]>>.
            
            classifier_model_names: List of names (string array) of classifiers in classifier_model_list.
            
            features_test: Input features for classification, classifier.predict_proba(features_test).
            
            labels_test: Correct labels for test data (features_test).
            
            n_bins: Number of bins for calibration plot.
        
        Output: Plots a calibartion plot displaying the observed occurrence rates vs the bin 
            midpoints for each mode
    """

    # List of values of y for each x for calibration curves of each model.
    calibration_curve_x_y = []

    # Fill calibration_curves_x_y
    for classifier in classifier_model_list:
        predicted_probablities = classifier.predict_proba(features_test)

        y, x = calibration_curve(
            labels_test, predicted_probablities[:, 1], n_bins=n_bins
        )

        calibration_curve_x_y.append([x, y])

    # Plot calibaration curve comparing different models in classifier_model_list.
    fig, ax = plt.subplots()

    for i in range(len(calibration_curve_x_y)):
        plt.plot(
            calibration_curve_x_y[i][0],
            calibration_curve_x_y[i][1],
            marker="o",
            linewidth=1,
            label=classifier_model_names[i],
        )

    # Reference line (y=x)
    line = mlines.Line2D([0, 1], [0, 1], color="black")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)

    # Title, legend, and axis labels
    fig.suptitle("Calibration plots")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability in each bin")
    plt.legend()
    plt.show()
