import numpy as np
import matplotlib.pyplot as plt


def plot_misclassified_probablities(model_res, class_labels, correct_y_labels):
    """
        Input:
            model_res: Model result (n X k) matrix where n = number of test_examples, k = number of classes.
                Each row has probablity values predicted by the model for each of k classes.
            class_lables: Labels for classes (1,2,3,...k) in order according to model_res.
            correct_y: Array of n values with each value as label for classes.
        Output:
            Plot showing classification probablities for misclassified data points.
    """

    # convert both input to numpy
    model_res = np.array(model_res)
    correct_y_labels = np.array(correct_y_labels)

    # create array correct_y to map correct_y_labels to corresponding index of class_labels
    correct_y = []
    for label in correct_y_labels:
        correct_y.append((np.where(class_labels == label))[0][0])

    # k = number of classes
    k = len(model_res[0])

    # indices in [0,n] of misclassified points
    index_misclassification = []

    # multi-class classification (predicted class is one with highest probablity)
    if k > 2:
        for res_i in range(len(model_res)):
            res = model_res[res_i]
            max_i = np.where(res == np.amax(res))
            if (len(max_i) > 1) or (max_i[0][0] != correct_y[res_i]):
                index_misclassification.append(res_i)
    elif k == 2:
        for res_i in range(len(model_res)):
            res = model_res[res_i]
            if not (res[correct_y[res_i]] > 0.5):
                index_misclassification.append(res_i)
    else:
        print("Invalid input.\nNumber of classes must be greater than pr equal to 2.")
        return

    # misclassification indices
    print("Number of misclassified datapoints: {}".format(len(index_misclassification)))

    # plotting data
    fig, ax = plt.subplots()

    # x used as index to show misclassifications
    x = np.arange(len(index_misclassification))
    plt.xticks(x, index_misclassification)

    x_shift = x - ((k - 1) * 0.25) - 0.125

    for i in range(k):
        label = "Class {}".format(class_labels[i])
        misc_probabs = []

        for j in range(len(index_misclassification)):
            misc_probabs.append(model_res[index_misclassification[j]][i])

        plt.bar(
            x_shift + (0.25 * i), misc_probabs, width=0.25, label=label,
        )

    ax.set_xlabel("Indices of misclassifications in Model Result.")
    ax.set_ylabel("Classification probablities.")

    title = "Misclassified data_point probablities for various classes."
    fig.suptitle(title, fontweight="bold")

    plt.legend()
    plt.show()
