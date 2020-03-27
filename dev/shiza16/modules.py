import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score


def dataset_statistics(data):

    """ Data Features and Labels are extracted """

    print("Shape of the dataset: ", data.shape)
    print("\nFeatures of the dataset are: \n", list(data.keys()))
    print("\nTarget Labels of dataset are: \n", data["Class"].unique())


def BarChart(dataa):
    """ BarChart for analyzing the frequency of Categorical labled class """

    print("\nBarChart for analyzing the frequency of Categorical labled class.\n")
    base_color = sns.color_palette()[9]
    Vorder = dataa["Class"].value_counts().index
    sns.countplot(data=dataa, x="Class", color=base_color, order=Vorder)


def Correlation_matrix(dataa):
    """ Correlation matrix to find the relationship between variables """

    print("Correlation Analysis\n")
    plt.figure(figsize=(25, 15))
    sns.heatmap(dataa.corr(), annot=True, linewidths=0.5)
    plt.show()


def label_encoding(lencoder, vehicle):
    """ Converting categorical labels into numeric values """

    vdataset = vehicle.copy()
    vdataset["Class_code"] = lencoder.fit_transform(vehicle["Class"])
    return vdataset


def splitting_train_test_data(data, size):
    """ 
    Data is splitted into ratio of size for training and testing
    
    """

    X = data.drop(["Class", "Class_code"], axis=1)
    y = data["Class_code"]

    return train_test_split(X, y, test_size=size, random_state=45)


def LogisticRegression_train(X, y):
    """ Logistic Regression Classifier"""

    classifier = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=7600)
    return classifier.fit(X, y)


def test_classifier(classifier, X_test):

    """ Evaluating model by predicting on testing data """

    y_predict = classifier.predict(X_test)
    return y_predict


def cross_validation(dataa, classifier):
    """ Cross Validation for performance evaluation  """

    X = dataa.drop(["Class", "Class_code"], axis=1)
    y = dataa[["Class_code"]]
    scores = cross_val_score(classifier, X, y, cv=5, scoring="accuracy")
    return scores.mean()


def model_confusion_matrix(y_test, y_predict, dataa):
    """ Drawing Confusion Matrix """

    fig = plt.gcf()
    fig.set_size_inches(8, 5)

    target = dataa["Class"].unique()  ##for index labels

    matrix = confusion_matrix(y_test, y_predict, labels=target)
    cmatrix = pd.DataFrame(matrix, index=target, columns=target)
    sns.heatmap(cmatrix, annot=True, linewidths=0.5)

    plt.title("Confusion Matrix for Logistic Regression \n")
    plt.ylabel("Actual Labels\n")
    plt.xlabel("\nPredicted Labels")
    plt.tight_layout()

    plt.tight_layout()

    return matrix


def model_classification_report(y_test, y_predict):
    """  Model Classification report for Precision , Recall and F1-Score """

    print("\nDataSet Report: ")
    print(classification_report(y_test, y_predict))


def randomize_data(X, Y):
    """ Randomize the labels and features data for learning curve"""

    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = Y[permutation]
    return X2, Y2
