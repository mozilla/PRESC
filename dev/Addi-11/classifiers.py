""" This file contains various classifiers to be used on the dataset """
from evaluation import evaluate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_precision_recall_curve, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt

class Classifier:

    def svm_classifier(self,X_train,y_train):
        classifier = SVC(gamma='auto')
        classifier.fit(X_train, y_train)
        return classifier

    def KNeighbors(self, X_train,y_train):
        classifier = KNeighborsClassifier()
        classifier.fit(X_train, y_train)
        return classifier

    def Logistic_Reg(self, X_train,y_train):
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        return classifier

    def Decision_Tree(self,X_train,y_train):
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def Random_Forest(self, X_train,y_train):
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def Gaussian(self, X_train,y_train):
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        return classifier

    """ This function is to evaluate classifier's performancen"""
    def evaluation(self, classifier, X_test, y_test):
        accuracy, precision, recall, f_score , y_score = evaluate(classifier, X_test, y_test)
        print("Accuracy : ",accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score : ",f_score)
        print("Precision vs Recall Curve")
        disp = plot_precision_recall_curve(classifier,X_test, y_test)
        print("Confusion Matrix")
        disp = plot_confusion_matrix(classifier,X_test, y_test)

    
        