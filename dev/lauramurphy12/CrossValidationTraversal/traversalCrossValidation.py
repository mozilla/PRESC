import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, fbeta_score, recall_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


def cv_space_traversal(ml_clf, data_features, data_labels, fold_num):
	k_folds = np.arange(2, fold_num)
	strat_repeated_k_folds = []
	stratkfold_accuracy_scores = []
	average_accuracies_stratkfold = []
	stratkfold_precision_scores = []
	stratkfold_average_precision = []
	stratkfold_fbeta_scores = []
	stratkfold_average_fbeta = []
	stratkfold_recall_scores = []
	stratkfold_average_recall = []
	for fold in k_folds:
		# Repeated Stratified K Fold repeated for given fold number. Set to 2 repetitions for each split
		rk = RepeatedStratifiedKFold(n_splits=fold, n_repeats=2)
		strat_repeated_k_folds.append(fold)
		for train_split, test_split in rk.split(data_features, data_labels):
			X_train, y_train = data_features[train_split], data_labels[train_split]
			X_test, y_test = data_features[test_split], data_labels[test_split]
			train_minority, test_minority = len(y_train[y_train == 1]), len(y_test[y_test == 1])
			train_majority, test_majority = len(y_train[y_train == 0]), len(y_test[y_test == 0])
			ml_clf.fit(X_train, y_train)
			y_pred = ml_clf.predict(X_test)
			stratkfold_accuracy_scores.append(round(accuracy_score(y_test, y_pred), 3))
			stratkfold_precision_scores.append(round(precision_score(y_test, y_pred), 3))
			stratkfold_recall_scores.append(round(recall_score(y_test, y_pred), 3))
			stratkfold_fbeta_scores.append(round(fbeta_score(y_test, y_pred, beta=0.3), 3))
		
		average_accuracy = round(sum(stratkfold_accuracy_scores)/len(stratkfold_accuracy_scores),3)
		average_accuracies_stratkfold.append(average_accuracy)
		average_precision = round(sum(stratkfold_precision_scores)/len(stratkfold_precision_scores),3)
		stratkfold_average_precision.append(average_precision)
		average_recall = round(sum(stratkfold_recall_scores)/len(stratkfold_recall_scores),3)
		stratkfold_average_recall.append(average_recall)
		average_fbeta = round(sum(stratkfold_fbeta_scores)/len(stratkfold_fbeta_scores),3)
		stratkfold_average_fbeta.append(average_fbeta)
	return(strat_repeated_k_folds, average_accuracies_stratkfold, stratkfold_average_precision, stratkfold_average_recall,  stratkfold_average_fbeta)

	

def cross_val_table(num_folds, accuracy_vals, precision_vals, recall_vals, fbeta_vals):
	cross_val_data = {'Folds': num_folds, 'Average Accuracy Score': accuracy_vals, 'Average Precision Score': precision_vals,'Average Recall Score': recall_vals, 'Average fbeta Score': fbeta_vals}
	scores_table = pd.DataFrame(cross_val_data, columns = ['Folds', 'Average Accuracy Score', 'Average Precision Score', 'Average Recall Score', 'Average fbeta Score'])
	return scores_table


def accuracy_lineplot_graph(dtf):
	sns.lineplot(x="Folds", y="Average Accuracy Score", data=dtf)

	
def precision_lineplot_graph(dtf):
	sns.lineplot(x="Folds", y="Average Precision Score", data=dtf)
	

def recall_lineplot_graph(dtf):
	sns.lineplot(x="Folds", y="Average Recall Score", data=dtf)

def fbeta_lineplot_graph(dtf):
	sns.lineplot(x="Folds", y="Average fbeta Score", data=dtf)



