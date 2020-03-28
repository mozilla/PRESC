
# This file contains various classifiers to be used on the dataset
from evaluation import evaluate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# from sklearn.metrics import (
# 	plot_precision_recall_curve,
# 	plot_confusion_matrix,
# )

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Classifier:
    """
	This class contains different classification models which can be trained on the dataset.
	"""

    def svm_classifier(self, x_train, y_train):
        """
		Support Vector Machine is a classifier
		Parameters :
			x_train : array-like, shape (n_samples, n_features)
			y_train : of length n_samples

		Returns :
			classifier : trained classification model 
		"""
        classifier = SVC(gamma="auto")
        classifier.fit(x_train, y_train)
        return classifier
    def KNeighbors(self, x_train, y_train):
        """
		K-Nearest Neighbours is supervised classifier, which takes a bunch of labelled points and uses them to learn how to label other points, wrt to thier degree of closeness.

		Parameters :
			x_train : array-like, shape (n_samples, n_features)
			y_train : of length n_samples

		Returns :
			classifier : trained classification model
		"""
        classifier = KNeighborsClassifier()
        classifier.fit(x_train, y_train)
        return classifier
    def Logistic_Reg(self, x_train, y_train):
        """
		Logistic Regression, takes some input and calculates the probabilty of the outcome using mathematical functions like sigmoid or ReLu.

		Parameters :
			x_train : array-like, shape (n_samples, n_features)
			y_train : of length n_samples

		Returns :
			classifier : trained classification model
		"""
        classifier = LogisticRegression()
        classifier.fit(x_train, y_train)
        return classifier
    def Decision_Tree(self, x_train, y_train):
        """
		Decision Tree Classifier, a mechanical way to make a decision by dividing the inputs into smaller decisions.

		Parameters :
			x_train : array-like, shape (n_samples, n_features)
			y_train : of length n_samples

		Returns :
			classifier : trained classification model
		"""
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train, y_train)
        return classifier
    def Random_Forest(self, x_train, y_train):
        """
		Random Forest Classifier, a way to make a decision by dividing the inputs into smaller decisions, with some randomness.The group outcomes are based on the positive responses. Used in Recommendation Systems.

		Parameters :
			x_train : array-like, shape (n_samples, n_features)
			y_train : of length n_samples

		Returns :
			classifier : trained classification model
		"""
        classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        return classifier
    def Gaussian(self, x_train, y_train):
        """
		Gaussian Naive Bayes, classification technique based on Bayes’ Theorem with an assumption of independence among predictors. It is easy to build and particularly useful for very large data sets.
		
		Parameters :
			x_train : array-like, shape (n_samples, n_features)
			y_train : of length n_samples

		Returns :
			classifier : trained classification model
		"""
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        return classifier
    def evaluation(self, classifier, x_val, y_val):
        """
		This function is used to evaluate the performance of the trained model, using evaluation metrics like :
			Accuracy
			Precision
			Recall
			Precision Recall Curve
			F1_score
			Confusion Matrix
			AUC-ROC Curve, on the validation set.
			
		Parameters :
			classifier : trained classification model
			x_val : array-like, shape(n_samples, n_features)
			y_val : of length n_samples

		Returns :
			void
		"""
        accuracy, precision, recall, f_score, y_score = evaluate(
            classifier, x_val, y_val
        )
        print("Accuracy : ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score : ", f_score)

        # Plotting Precision Recall Curve
        # print("Precision vs Recall Curve")
        # disp = plot_precision_recall_curve(classifier, x_val, y_val)

        # Plotting Confusion Matrix
        print("Confusion Matrix")
        labels = ["Class 1", "Class 2"]
        cm = confusion_matrix(y_val, y_score)
        print(cm)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title("Confusion matrix of the classifier")
        fig.colorbar(cax)
        ax.set_xticklabels([""] + labels)
        ax.set_yticklabels([""] + labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
