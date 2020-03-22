import numpy as np
import matplotlib.pyplot as plt


def plot_misclassified_probablities(
    model_res, class_labels, correct_y_labels, n_datapoints
):
    """
        Input:
            model_res: Model result (n X k) matrix where n = number of test_examples, k = number of classes.
                Each row has probablity values predicted by the model for each of k classes.
            class_lables: Labels for classes (1,2,3,...k) in order according to model_res.
            correct_y: Array of n values with each value as label for classes.
            n_datapoints: Number of datapoints you wan't to plot.
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
    print(
        "Total number of misclassified datapoints: {}".format(
            len(index_misclassification)
        )
    )

    # changing index_misclassification to contain first min(n_datapoints, len(index_misclassification)) points
    index_misclassification = index_misclassification[
        : min(n_datapoints, len(index_misclassification))
    ]

    # plotting data
    fig, ax = plt.subplots()

    # width of histogram bars
    width = 0.25
    step = (k + 2) * width
    colors = np.transpose(np.random.rand(3, k))

    # x used as index to show misclassifications
    x = np.arange(start=0, stop=(len(index_misclassification)) * step, step=step)
    plt.xticks(x, index_misclassification)

    x_shift = x - (((k / 2) - 1) * width) - (width / 2)

    for i in range(k):
        label = "Class {}".format(class_labels[i])
        misc_probabs = []

        for j in range(len(index_misclassification)):
            misc_probabs.append(model_res[index_misclassification[j]][i])

        plt.bar(
            x_shift + (width * i),
            misc_probabs,
            color=colors[i],
            width=width,
            label=label,
        )

    for i in range(len(index_misclassification)):
        correct_class = correct_y[index_misclassification[i]]
        plt.plot(x[i], 1, color=colors[correct_class], marker="o",)

    ax.set_xlabel("Indices of misclassifications in Model Result.")
    ax.set_ylabel("Classification probablities.")

    title = "Misclassified data_point probablities for various classes."
    fig.suptitle(title, fontweight="bold")

    plt.legend()
    plt.show()
