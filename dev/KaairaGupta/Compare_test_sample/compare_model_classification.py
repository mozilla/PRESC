import matplotlib.pyplot as plt
import numpy as np


def compare_model_classification(model1_res, model2_res, y_test):
    """
        This function plot visualization to compare predicted class probabilities across two binary-classifier models.
        Input:
            model1_res, model2_res: Each model's test samples prediction, i.e. arrays size (n*2), where n are the number of outputs. 
                The 2 columns are of the form [P(y=0), P(y=1)].
            y_test: The actual value of y (class) (either 0 or 1).
        Output: Two visualisations (one for each class, 0 and 1) for each model.
    """

    # table t contains probablity values given by both models for each model
    t = [[], []]

    for i in range(len(model1_res)):
        y = y_test[i]
        temp = [
            min(model1_res[i][y], round(model1_res[i][y])),
            min(model2_res[i][y], round(model2_res[i][y])),
        ]
        if not (temp[0] == 0 and temp[1] == 0):
            t[y].append(temp)

    t_trans = [np.array(t[0]).transpose(), np.array(t[1]).transpose()]

    for i in range(2):
        fig, ax = plt.subplots()

        # x used as index to show performance at various test examples
        x = np.arange(len(t[i]))
        plt.xticks(x)

        plt.bar(x - 0.125, t_trans[i][:][0], color="y", width=0.25, label="Model 1")
        plt.bar(
            x - 0.125 + 0.25, t_trans[i][:][1], color="g", width=0.25, label="Model 2"
        )

        ax.set_xlabel("Index: Test Examples")
        ax.set_ylabel("Probablity for correct class (y={}) prediction".format(i))

        title = "Performace for y = {} for two models.".format(i)
        fig.suptitle(title, fontweight="bold")

        plt.legend()
        plt.show()
