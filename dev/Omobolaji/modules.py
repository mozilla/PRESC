import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Different models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def statistics(data):
    """View Information on the Dataset"""
    print("STATISTICS \n")
    print("Shape of data: ", data.shape, " \n")
    print("Columns in Data: \n", data.columns, " \n")
    print("Categories of Target Column: \n", data.Class.unique(), " \n")
    print(
        "Number of each category in Target Column: \n", data.Class.value_counts(), " \n"
    )
    print("Total number of null values in each column: \n", data.isnull().sum(), " \n")
    print("Description of each continuous feature: \n", data.describe().T, " \n")

def visualization(data):
    """Graphical Information on the Dataset"""
    print("An histogram showing categories in Target column, Class \n")
    sns.countplot(data=data, x="Class")
    plt.show()

    print("Correlation Analysis of Vehicle dataset \n")
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    sns.heatmap(data.corr(), annot=True)
    plt.show()

def preprocessing(data):
    """Convert categorical column to continuous"""
    le = LabelEncoder()
    data["Class"] = le.fit_transform(data["Class"])
    """Split the columns into features and target"""
    features = data.drop(["Class"], axis=1)
    target = data.Class
    return train_test_split(features, target, test_size=0.3, random_state=42)

def logistic_regression_model(train_features, test_features, train_target, test_target):
    """Applying Logistic Regression model on the dataset"""
    lr = LogisticRegression(solver="liblinear", multi_class="ovr")
    lr.fit(train_features, train_target)
    predict_lr = lr.predict(test_features)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(
        lr, train_features, train_target, cv=kfold, scoring="accuracy"
    )
    cv = cv_results.mean(), cv_results.std()
    print("Cross Validation: ", cv)
    accuracy_score = metrics.accuracy_score(test_target, predict_lr)
    print("Accuracy Score(%): ", accuracy_score * 100)
    print(confusion_matrix(test_target, predict_lr))
    print(classification_report(test_target, predict_lr))

def k_neighbours_classifier_model(
    train_features, test_features, train_target, test_target
):
    """Applying KNeigbours Classifier model on the dataset"""
    knn = KNeighborsClassifier()
    knn.fit(train_features, train_target)
    predict_knn = knn.predict(test_features)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(
        knn, train_features, train_target, cv=kfold, scoring="accuracy"
    )
    cv = cv_results.mean(), cv_results.std()
    print("Cross Validation: ", cv)
    accuracy_score = metrics.accuracy_score(test_target, predict_knn)
    print("Accuracy Score(%): ", accuracy_score * 100)
    print(confusion_matrix(test_target, predict_knn))
    print(classification_report(test_target, predict_knn))

def randomforest_classifier_model(
    train_features, test_features, train_target, test_target
):
    """Applying Random Forest Classifier model on the dataset"""
    rand = RandomForestClassifier(n_estimators=10, criterion="entropy")
    rand.fit(train_features, train_target)
    predict_rand = rand.predict(test_features)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(
        rand, train_features, train_target, cv=kfold, scoring="accuracy"
    )
    cv = cv_results.mean(), cv_results.std()
    print("Cross Validation: ", cv)
    accuracy_score = metrics.accuracy_score(test_target, predict_rand)
    print("Accuracy Score(%): ", accuracy_score * 100)
    print(confusion_matrix(test_target, predict_rand))
    print(classification_report(test_target, predict_rand))

def decision_tree_classifier_model(
    train_features, test_features, train_target, test_target
):
    """Applying Decision Tree Classifier model on the dataset"""
    dtree = DecisionTreeClassifier()
    dtree.fit(train_features, train_target)
    predict_dtree = dtree.predict(test_features)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(
        dtree, train_features, train_target, cv=kfold, scoring="accuracy"
    )
    cv = cv_results.mean(), cv_results.std()
    print("Cross Validation: ", cv)
    accuracy_score = metrics.accuracy_score(test_target, predict_dtree)
    print("Accuracy Score(%): ", accuracy_score * 100)
    print(confusion_matrix(test_target, predict_dtree))
    print(classification_report(test_target, predict_dtree))

def linear_discriminant_analysis_model(
    train_features, test_features, train_target, test_target
):
    """Applying Linear Discriminant Analysis model on the dataset"""
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_features, train_target)
    predict_lda = lda.predict(test_features)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(
        lda, train_features, train_target, cv=kfold, scoring="accuracy"
    )
    cv = cv_results.mean(), cv_results.std()
    print("Cross Validation: ", cv)
    accuracy_score = metrics.accuracy_score(test_target, predict_lda)
    print("Accuracy Score(%): ", accuracy_score * 100)
    print(confusion_matrix(test_target, predict_lda))
    print(classification_report(test_target, predict_lda))

def gaussian_nb_model(train_features, test_features, train_target, test_target):
    """Applying Gaussian NB model on the dataset"""
    nb = GaussianNB()
    nb.fit(train_features, train_target)
    predict_nb = nb.predict(test_features)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(
        nb, train_features, train_target, cv=kfold, scoring="accuracy"
    )
    cv = cv_results.mean(), cv_results.std()
    print("Cross Validation: ", cv)
    accuracy_score = metrics.accuracy_score(test_target, predict_nb)
    print("Accuracy Score(%): ", accuracy_score * 100)
    print(confusion_matrix(test_target, predict_nb))
    print(classification_report(test_target, predict_nb))
