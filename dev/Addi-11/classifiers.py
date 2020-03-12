# This file contains various classifiers to be used on the dataset 
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
    '''
    This class contains different classification models which can be trained on the dataset.
    '''
    def svm_classifier(self,X_train,y_train):
        '''
        Support Vector Machine is a classifier
        Parameters :
            X_train : array-like, shape (n_samples, n_features)
            y_train : of length n_samples

        Returns :
            classifier : trained classification model 
        '''
        classifier = SVC(gamma='auto')
        classifier.fit(X_train, y_train)
        return classifier

    def KNeighbors(self, X_train,y_train):
        '''
        K-Nearest Neighbours is supervised classifier, which takes a bunch of labelled points and uses them to learn how to label other points, wrt to thier degree of closeness.

        Parameters :
            X_train : array-like, shape (n_samples, n_features)
            y_train : of length n_samples

        Returns :
            classifier : trained classification model
        '''
        classifier = KNeighborsClassifier()
        classifier.fit(X_train, y_train)
        return classifier

    def Logistic_Reg(self, X_train,y_train):
        '''
        Logistic Regression, takes some input and calculates the probabilty of the outcome using mathematical functions like sigmoid or ReLu.

        Parameters :
            X_train : array-like, shape (n_samples, n_features)
            y_train : of length n_samples

        Returns :
            classifier : trained classification model
        '''
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        return classifier

    def Decision_Tree(self,X_train,y_train):
        '''
        Decision Tree Classifier, a mechanical way to make a decision by dividing the inputs into smaller decisions.

        Parameters :
            X_train : array-like, shape (n_samples, n_features)
            y_train : of length n_samples

        Returns :
            classifier : trained classification model
        '''
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def Random_Forest(self, X_train,y_train):
        '''
        Random Forest Classifier, a way to make a decision by dividing the inputs into smaller decisions, with some randomness.The group outcomes are based on the positive responses. Used in Recommendation Systems.

        Parameters :
            X_train : array-like, shape (n_samples, n_features)
            y_train : of length n_samples

        Returns :
            classifier : trained classification model
        '''
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        return classifier
    
    def Gaussian(self, X_train,y_train):
        '''
        Gaussian Naive Bayes, classification technique based on Bayes’ Theorem with an assumption of independence among predictors. It is easy to build and particularly useful for very large data sets.
        
        Parameters :
            X_train : array-like, shape (n_samples, n_features)
            y_train : of length n_samples

        Returns :
            classifier : trained classification model
        '''
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        return classifier

    def evaluation(self, classifier, X_test, y_test):
        '''
        This function is used to evaluate the performance of the trained model, using evaluation metrics like :
            Accuracy
            Precision
            Recall
            Precision Recall Curve
            F1_score
            Confusion Matrix
            AUC-ROC Curve
            
        Parameters :
            classifier : trained classification model
            X_test : array-like, shape(n_samples, n_features)
            y_test : of length n_samples

        Returns :
            void
        '''
        accuracy, precision, recall, f_score , _ = evaluate(classifier, X_test, y_test)
        print("Accuracy : ",accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score : ",f_score)
        print("Precision vs Recall Curve")
        disp = plot_precision_recall_curve(classifier,X_test, y_test)
        print("Confusion Matrix")
        disp = plot_confusion_matrix(classifier,X_test, y_test)

    
        