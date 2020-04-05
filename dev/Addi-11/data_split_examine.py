# This file compares various evaluation metrics for different data splits 

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython.display import HTML
from pylab import *
from dataloader import get_x_y
from evaluation import evaluate
from classifiers import Classifier



test_sizes = np.arange(0.005,1,0.05)
columns = ['Training data','Testing Data','Accuracy %', 'Precision', 'Recall', 'F1_score']
df = pd.DataFrame(columns = columns)

def data_split_examine(clf):
	'''
	The fuction calculates evaluation metrics like f1_score, accuracy, precision, recall for various test data sizes

	Parameters:
		clf : a trained classification model

	Return:
		void
	'''
	model = Classifier()
	for index in range(len(test_sizes)):
		x, y = get_x_y() 
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_sizes[index])
		classifier = getattr(model, clf)(x_train, y_train) 
		accuracy, precision, recall, f_score, _ = evaluate(classifier, x_test, y_test)
		train = round((1-test_sizes[index])*100)
		test = round(test_sizes[index]*100)
		df.loc[index+1] = [train, test, accuracy*100, precision, recall, f_score]

	display(df)
		
def visualise_split(clf):
	'''
	The function visualises the corelation between data splits and evaluation metrics by plotting graph between testing data sizes and accuracy.
	'''
	fig,axes = plt.subplots()
	axes.set_xlabel("Test Data Size")
	axes.set_ylabel("Accuracy %")
	axes.set_ylim([50,100])
	axes.set_title("Relation btw accuracy and test data size for {} classifier".format(clf))
	disp = axes.plot(df['Testing Data'], df['Accuracy %'])
