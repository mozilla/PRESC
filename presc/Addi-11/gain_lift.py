import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np

def gain_chart(clf, x_val, y_val):
	'''
	Gain and Lift Charts visualise how much better one can except to do with the predictive model comparing without a model.

	Parameters:
		clf : trained classification model
		x_val : array-like, shape(n_samples, n_features)
		y_val : of length n_features

	Returns:
		null
	'''
	y_score = clf.predict_proba(x_val)
	skplt.metrics.plot_cumulative_gain(y_val, y_score)
	plt.show()

def lift_chart(clf, x_val, y_val):
	# model's output prediction
	y_score = clf.predict(x_val)
	# dataframe holding both actual and predicted values
	df_dict = {'actual': list (y_val), 'pred': list(y_score)}
	df = pd.DataFrame(df_dict)
	# creating equal bins of size 100 by division using ranking system
	pred_ranks = pd.qcut(df['pred'].rank(method='first'), 100, labels=False)
	actual_ranks = pd.qcut(df['actual'].rank(method='first'), 100, labels=False)
	pred_percentiles = df.groupby(pred_ranks).mean()
	actual_percentiles = df.groupby(actual_ranks).mean()
	plt.title('Lift Chart')
	plt.plot(np.arange(.01, 1.01, .01), np.array(pred_percentiles['pred']),color='darkorange', lw=2, label='Prediction')
	plt.plot(np.arange(.01, 1.01, .01), np.array(pred_percentiles['actual']),color='navy', lw=2, linestyle='--', label='Actual')
	plt.ylabel('Class Predicted')
	plt.xlabel('Percentage of Sample')
	plt.legend(loc="best")


