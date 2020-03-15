import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
<<<<<<< HEAD
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score
from sklearn import metrics
=======
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530
from sklearn.model_selection import cross_val_score


def dataset_statistics(data):

    """ Data Features and Labels"""

    print("Shape of the dataset: ", data.shape)
<<<<<<< HEAD

    print("\nFeatures of the dataset are: ", list(data.keys()))

    print("\nTarget Labels of dataset are: ", data["Class"].unique())


def data_visualization(dataa):

    """ For Visulaization of DataSet"""
=======
    print("\nFeatures of the dataset are: \n", list(data.keys()))
    print("\nTarget Labels of dataset are: \n", data["Class"].unique())


def histogram(dataa):
    """ Histogram for analyzing the frequency of labled class """
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530

    print("\nHistogram for analyzing the frequency of labled class.\n")
    base_color = sns.color_palette()[9]
    Vorder = dataa["Class"].value_counts().index
    sns.countplot(data=dataa, x="Class", color=base_color, order=Vorder)
<<<<<<< HEAD

    print("\n")
=======


def Correlation_matrix(dataa):
    """ Correlation matrix to find the relationship between variables """

    print("Correlation Analysis\n")
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530
    plt.figure(figsize=(25, 15))
    sns.heatmap(dataa.corr(), annot=True, linewidths=0.5)
    plt.show()
    print("Correlation Analysis.")


<<<<<<< HEAD
def splitting_train_test_data(data):

    """ Data is splitted into 30:70 for training and testing"""
    # X = data.drop(['Class'] , axis = 1)
    # y  = data['Class']
    X = data.drop(["18"], axis=1)
    y = data[["18"]]

    return train_test_split(X, y, test_size=0.3, random_state=45)


def SVM_train(X, y):
=======

def label_encoding(vehicle):
    """ Converting categorical labels into numeric values """

    vdataset = vehicle.copy()
    from sklearn.preprocessing import LabelEncoder

    lb_make = LabelEncoder()
    vdataset["Class_code"] = lb_make.fit_transform(vehicle["Class"])
    return vdataset


def splitting_train_test_data(data):
    """ Data is splitted into 30:70 for training and testing"""

    X = data.drop(["Class", "Class_code"], axis=1)
    y = data["Class_code"]

    return train_test_split(X, y, test_size=0.3, random_state=45)


def LogisticRegression_train(X, y):
    """ Logistic Regression Classifier"""

    classifier = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=7600)
    return classifier.fit(X, y)
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530

    """ SVM Classifier"""
    # kernel =' poly ' is taking infinte time that's why it is not added.
    params_grid = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]}]

<<<<<<< HEAD
    svm_grid = GridSearchCV(SVC(), params_grid, cv=5)
    svm_grid = svm_grid.fit(X, y)
    classifier = SVC(
        kernel=svm_grid.best_estimator_.kernel,
        C=svm_grid.best_estimator_.C,
        gamma=svm_grid.best_estimator_.gamma,
    )

    return classifier.fit(X, y)


def LogisticRegression_train(X, y):

    """ SVM Classifier"""
    classifier = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=7600)
    return classifier.fit(X, y)


def Tuning_LogiticRegression(X, y, x):
    """ Tuning Logistic Regression to increase the accuracy of model """

    parameters = {
        "penalty": ["l1", "l2"],
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "solver": ["liblinear", "saga"],
        "class_weight": [
            {1: 0.5, 0: 0.5},
            {1: 0.4, 0: 0.6},
            {1: 0.6, 0: 0.4},
            {1: 0.7, 0: 0.3},
        ],
    }
    LR_grid = GridSearchCV(LogisticRegression(), param_grid=parameters, cv=x)
    LR_grid = LR_grid.fit(X, y)
    classifier = LogisticRegression(
        penalty=LR_grid.best_estimator_.get_params()["penalty"],
        C=LR_grid.best_estimator_.get_params()["C"],
    )
    return classifier.fit(X, y)


=======
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530
def test_classifier(classifier, X_test):

    """ Evaluating model by predicting on testing data """

    y_predict = classifier.predict(X_test)
    return y_predict


<<<<<<< HEAD
def cross_validation(X, y, classifier):
    scores = cross_val_score(classifier, X, y, cv=5, scoring="f1_macro")
=======
def cross_validation(dataa, classifier):
    """ Cross Validation for performance evaluation  """

    X = dataa.drop(["Class", "Class_code"], axis=1)
    y = dataa[["Class_code"]]
    scores = cross_val_score(classifier, X, y, cv=5, scoring="accuracy")
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530
    return scores.mean()


def model_confusion_matrix(y_test, y_predict, dataa):
<<<<<<< HEAD

=======
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530
    """ Drawing Confusion Matrix """

    fig = plt.gcf()
    fig.set_size_inches(8, 5)

    target_label = dataa["Class_code"].unique()
<<<<<<< HEAD
    target = dataa["Class"].unique()
=======
    target = dataa["Class"].unique()  ##for index labels
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530

    matrix = confusion_matrix(y_test, y_predict, labels=target_label)
    cmatrix = pd.DataFrame(matrix, index=target, columns=target)
    sns.heatmap(cmatrix, annot=True, linewidths=0.5)

    plt.title("Confusion Matrix for Logistic Regression \n")
<<<<<<< HEAD
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
=======
    plt.ylabel("Actual Labels\n")
    plt.xlabel("\nPredicted Labels")
    plt.tight_layout()

>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530
    plt.tight_layout()


def model_classification_report(y_test, y_predict):

    """  Model Classification report for Precision , Recall and F1-Score """

    print("\nDataSet Report: ")
    print(classification_report(y_test, y_predict))


def randomize_data(X, Y):
    """ Randomize the labels and features data for learning curve"""

    permutation = np.random.permutation(Y.shape[0])
    X2 = X[permutation, :]
    Y2 = Y[permutation]
<<<<<<< HEAD
    return X2, Y2


def draw_learning_curves(X, y, estimator, num_trainings):

    """ Method to draw learning curves of different models """

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=None,
        n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, num_trainings),
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()

    plt.title(" Learning Curves ")
    plt.xlabel(" Training Score ")
    plt.ylabel(" Score ")

    plt.plot(train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(test_scores_mean, "o-", color="b", label="Cross-validation score")

    plt.legend(loc="best")

    plt.show()
=======
    return X2, Y2
>>>>>>> 41f004f8e0047fd08441753984fb63d5a62c5530
