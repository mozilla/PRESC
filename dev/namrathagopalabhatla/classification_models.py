import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from data_processing import dim_reduction

from seaborn import heatmap
from IPython.display import display


class Classifier:
    def __init__(
        self,
        model="logistic",
        hidden_size=None,
        activation="relu",
        solver="lbfgs",
        max_iter=1000,
        multi_class="auto",
        gamma='auto',
        kernel='rbf',
        degree=3,
        k=5
    ):
        """Constructor for initalization of model using provided parameters"""
        self.model = model
        if self.model == "mlp":
            if hidden_size is None:
                print("Hidden layer(s)' size(s) not specified.")
                return None

            self.classifier = MLPClassifier(
                hidden_size, solver=solver, activation=activation, max_iter=max_iter
            )

        elif self.model == "logistic":
            self.classifier = LogisticRegression(solver=solver, multi_class=multi_class)
        
        elif self.model == "svm":
            self.classifier = SVC(kernel=kernel, degree=degree, gamma=gamma, probability=True)

        elif self.model == "knn":
            self.classifier = KNeighborsClassifier(n_neighbors=k)

        else:
            self.model = "logistic"
            self.classifier = LogisticRegression(solver=solver, multi_class=multi_class)

        self.predictions = None
        self.probabilities = None
        self.report = None
        self.confusion = None
        self.roc = None
        self.auroc = None
        self.accuracy = None
        self.f_score = None

    def classify(self, X=None):
        """Function that returns prediction based on trained model"""
        if X is None:
            print("Data not provided.")
            return None

        prediction = self.classifier.predict(X)
        return prediction

    def train(self, X=None, y=None):
        """Function to train the classification model"""
        if X is None or y is None:
            print("Data not provided.")
            return False

        self.classifier.fit(X, y)
        return True

    def confusion_matrix(self, X=None, y=None):
        """Function to show and return confusion matrix"""
        if self.confusion is None and X is None and y is None:
            print("Data not provided.")
            return None
        elif self.confusion is None and X is not None and y is not None:
            _, _ = self.validate(X, y)

        ax = plt.axes()
        heatmap(self.confusion, annot=True, ax=ax)
        ax.set_title("Confusion Matrix")
        plt.show()

        return self.confusion

    def show_report(self, X=None, y=None):
        """Function to show and return classification report including accuracy, F1 score, precision, recall, etc."""
        if self.report is None and X is None and y is None:
            print("Data not provided.")
            return None
        elif self.report is None and X is not None and y is not None:
            _, _ = self.validate(X, y)

        display(self.report)
        return self.report

    def show_misclassification(self, X=None, y=None):
        """Function to show True Positives and False Positives of each class"""
        if X is None or y is None:
            print("Data not provided.")
            return None

        predictions = self.classifier.predict(X)
        unique_classes = np.unique(y)

        LDA = dim_reduction("lda", X, y, 2)

        for unique_class in unique_classes:
            all_positives = predictions == unique_class
            true_values = y == unique_class
            true_positives = true_values * all_positives
            false_positives = np.bitwise_not(true_values) * all_positives

            plt.figure(figsize=(8, 8))
            plt.scatter(
                LDA[true_positives][:, 0],
                LDA[true_positives][:, 1],
                c="green",
                label="True Positives",
            )
            plt.scatter(
                LDA[false_positives][:, 0],
                LDA[false_positives][:, 1],
                c="red",
                label="False Positives",
            )
            plt.legend()
            plt.title("Class {}".format(unique_class))
            plt.show()

        return

    def AUROC(self, X=None, y=None):
        """Function to show ROC curve and return AUROC score"""
        if self.roc is None and self.auroc is None and (X is None or y is None):
            print("Data not provided.")
            return None
        elif (
            self.roc is None
            and self.auroc is None
            and (X is not None and y is not None)
        ):
            _, _ = self.validate(X, y)

        plt.figure(figsize=(10, 10))

        for i in range(len(self.roc)):
            plt.plot(
                self.roc[i][0],
                self.roc[i][1],
                label="Class {} vs Rest".format(self.roc[i][2]),
            )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

        for i in range(len(self.roc)):
            print(
                "AUROC Score for Class {0}: {1}".format(self.roc[i][2], self.auroc[i])
            )

        return self.auroc

    def model_accuracy(self, X=None, y=None):
        """Function to return accuracy of the model"""
        if self.accuracy is None and (X is None or y is None):
            print("Data not provided.")
            return None
        elif self.accuracy is None and (X is not None and y is not None):
            _, _ = self.validate(X, y)

        return self.accuracy

    def model_score(self, X=None, y=None):
        """Function to return F-score of the model"""
        if self.f_score is None and (X is None or y is None):
            print("Data not provided.")
            return None
        elif self.f_score is None and (X is not None and y is not None):
            _, _ = self.validate(X, y)

        return self.f_score
    
    def prediction_probabilities(self, X=None, y=None):
        """Function to return F-score of the model"""
        if self.probabilities is None and (X is None or y is None):
            print("Data not provided.")
            return None
        elif self.f_score is None and (X is not None and y is not None):
            _, _ = self.validate(X, y)

        return self.probabilities

    def validate(self, X, y):
        """Function to validate test/validation set and evaluate accuracy"""
        if X is None or y is None:
            print("Data not provided.")
            return None, None

        self.predictions = self.classify(X)
        probability = self.classifier.predict_proba(X)
        self.probabilities = np.copy(probability)

        self.report = classification_report(y, self.predictions, output_dict=True)
        self.report = pd.DataFrame(self.report).transpose()
        self.accuracy = accuracy_score(y, self.predictions)
        self.f_score = f1_score(y, self.predictions, average="weighted")

        self.confusion = confusion_matrix(y, self.predictions)

        unique_classes = np.unique(y)

        self.roc = []
        self.auroc = []

        for i in range(len(unique_classes)):
            scores = probability
            labels = np.copy(y)
            labels[labels == unique_classes[i]] = -1
            labels[labels != -1] = 1
            fpr, tpr, _ = roc_curve(y_true=labels, y_score=scores[:, i], pos_label=-1)
            self.roc.append((fpr, tpr, unique_classes[i]))
            self.auroc.append(auc(fpr, tpr))


        return self.report, self.confusion