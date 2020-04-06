import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def Misclasssification_visualization(y_test, y_predict, data):
    """  
    Visualizing Bar graph for Missclassification, values
    that are incorreclty predicted by a classifier.

    """

    target_label = data["recommend_code"].unique()
    label = data["recommend"].unique()
    cmatrix = pd.DataFrame(
        data=confusion_matrix(y_test, y_predict, labels=target_label),
        columns=label,
        index=label,
    )

    # Because we only want to print missclassification and all the diagonal
    # values are true classified values

    for label in cmatrix.columns:
        cmatrix.at[label, label] = 0

    graph = cmatrix.plot(
        kind="bar", title="Visualization of Misclassification", stacked=True
    )
    graph.set_xlabel("Class Labels")
    graph.set_ylabel("No of Misclassified points")
    plt.show()
    plt.show()
