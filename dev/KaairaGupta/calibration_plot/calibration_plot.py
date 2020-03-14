from sklearn.calibration import calibration_curve as calib_c
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


def calibration_plot(clf_list, clf_names, test_data, labels_test, n_bins):
    """
        Input:
            clf_list: list of objects of a classifier which has method clf.predict_proba(test_data) with output y = list of predicted prbablities for two classes <np.array<np.array[2]>>
            test_data: input for clf.predict(test_data)
            test_data_labels: correct labels for test data
            n_bins: number of bins
            clf_names: list of names clf in clf_list
        Output: Plots a calibartion plot displaying the observed occurrence rates vs the bin midpoints for each mode
    """

    cc_list = []

    for clf in clf_list:
        pred_p = clf.predict_proba(test_data)
        y, x = calib_c(labels_test, pred_p[:, 1], n_bins=10)
        cc_list.append([x, y])

    fig, ax = plt.subplots()

    for i in range(len(cc_list)):
        plt.plot(
            cc_list[i][0], cc_list[i][1], marker="o", linewidth=1, label=clf_names[i]
        )

    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color="black")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle("Calibration plots")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability in each bin")
    plt.legend()
    plt.show()
