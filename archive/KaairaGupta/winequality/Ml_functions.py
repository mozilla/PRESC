from data_loader import load_split_preprocessed_data
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_recall_curve,
    auc,
    roc_curve,
    roc_auc_score,
    classification_report,
)
import warnings
import matplotlib.pyplot as plt  # to plot graph
import seaborn as sns

warnings.filterwarnings("ignore")


def model(model, features_train, features_test, labels_train, labels_test):
    """
            Input: This function takes input in the form of a model, features_train: features which are to be trained, features_test: features which are to be tested, the labels of training data, and the labels of test data. 
            Output: This outputs various plots and confusion matrix for each model and data to help us understand which model to use and which gives highest accuracy/ F1 score.

    """

    clf = model
    clf.fit(features_train, labels_train.values.ravel())
    pred = clf.predict(features_test)
    cnf_matrix = confusion_matrix(labels_test, pred)
    print(
        "the recall for this model is :",
        cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]),
    )
    fig = plt.figure(figsize=(6, 3))  # to plot the graph
    print(
        "TP", cnf_matrix[1, 1,]
    )  # no of false recommendation which are predicted false
    print("TN", cnf_matrix[0, 0])  # no. of true recommendation which are predited true
    print("FP", cnf_matrix[0, 1])  # no of true recommendation which are predicted false
    print("FN", cnf_matrix[1, 0])  # no of false recommendation which are predicted true
    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test, pred))
