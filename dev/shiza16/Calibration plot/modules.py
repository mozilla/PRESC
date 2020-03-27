import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


lencoder = LabelEncoder()


def dataset_statistics(data):

    """ Data Features and Labels are extracted """

    print("Shape of the dataset: ", data.shape)
    print("\nFeatures of the dataset are: \n", list(data.keys()))
    print("\nTarget Labels of dataset are: \n", data["recommend"].unique())


def BarChart(dataa):
    """ BarChart for analyzing the frequency of Categorical labled class """

    print("\nBarChart for analyzing the frequency of Categorical labled class.\n")
    base_color = sns.color_palette()[9]
    Vorder = dataa.iloc[:, (len(dataa.columns) - 1)].value_counts().index
    sns.countplot(data=dataa, x="recommend", color=base_color, order=Vorder)


def Correlation_matrix(dataa):
    """ Correlation matrix to find the relationship between variables """

    print("Correlation Analysis\n")
    plt.figure(figsize=(25, 15))
    sns.heatmap(dataa.corr(), annot=True, linewidths=0.5)
    plt.show()


def label_encoding(vehicle):
    """ Converting categorical labels into numeric values """

    vdataset = vehicle.copy()
    vdataset["recommend_code"] = lencoder.fit_transform(vehicle["recommend"])
    return vdataset


def Oversampling(X_train, y_train):
    
    """
    Resampling data by using oversamplint technique to handle the imbalance data
    
    """
    
    data = pd.concat([X_train, y_train], axis=1)
    false = data[data.recommend_code==0]
    true = data[data.recommend_code==1]
    
    true_sampled = resample(true,
                          replace=True, 
                          n_samples=len(false), 
                          random_state=27)
 

    return pd.concat([false, true_sampled])

def splitting_train_test_data(data):
    """ Data is splitted into 30:70 for training and testing"""

    X = data.drop(["recommend", "recommend_code"], axis=1)
    y = data["recommend_code"]

    return train_test_split(X, y, test_size=0.2, random_state=45)


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


def model_confusion_matrix(y_test, y_predict, data):
    """ Drawing Confusion Matrix """

    target_label = data["recommend_code"].unique()
    target = data["recommend"].unique()  ##for index labels

    matrix = confusion_matrix(y_test, y_predict, labels=target_label)
    cmatrix = pd.DataFrame(matrix, index=target, columns=target)
    sns.heatmap(cmatrix, annot=True, linewidths=0.5)

    plt.title("Confusion Matrix for Logistic Regression \n")
    plt.ylabel("Actual Labels\n")
    plt.xlabel("\nPredicted Labels")
    plt.show()

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
