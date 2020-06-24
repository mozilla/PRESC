from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score,
    f1_score,
    make_scorer,
    recall_score,
    accuracy_score,
)
from numpy import mean


def define_metrics():
    """
	This method creates a scorer for all the important metrics like precison, recall, accuracy and f1 score 
	so during K Fold validation all the selected metrics can be scored/calculated simultaneously
	It is internally invoked by the 'KNNevaluate' method and not called by the user directly 

	It retruns a dictoinary, with the name of the metrics as key 
	"""
    metrics = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1_score": make_scorer(f1_score),
    }
    return metrics


def KNN(neighbors=3):
    """
	This method creates a model for the KNN classifier. It taken the number of neighbors considered by the classifier in 
	decision making as the parameter with the default value as 3. 
	For this classifier, using odd number of neighbors is recommended. 
	Though it is internally invoked by the 'KNNevaluate' method; its input parameter can be set using the 'neighbors' parameter 
	of the 'KNNevaluate' method 
	
	It retruns the KNeighbors Classifier model to 'KNNevaluate' method
	 
	"""
    model = KNeighborsClassifier(neighbors)
    return model


def KNNevaluate(feature_values, target, neighbors=3, splits=10):
    """
	This method takes the data/feature values ie the values of independent variables as input along with the target/label/dependent variable value
	It as optional paramters of neighbors which controls the number of neighbors considered while perfomring KNN algorithm and the number of splits 
	for cross validation
	The default values of neighbors is 3 and splits is 10 as those values gave optimal results 

	It returns a dictionary of scored metrics based on the feature values passed which serve as the input to other methods 
	"""
    kf = KFold(n_splits=splits, shuffle=True)
    model = KNN(neighbors)
    scores = cross_validate(
        estimator=model, X=feature_values, y=target, cv=kf, scoring=define_metrics()
    )
    return scores


def view_average_metrics(scores):
    """
	This method takes a dictionary outputed by the 'KNNevaluate' method as its input

	It returns avarage values of the evaluation metrics based on the data passed to the 'KNNevaluate' method  
	"""
    print(
        " Average accuracy {:.2f} \n Average precision {:.2f} \n Average recall {:.2f} \n Average f1 score {:.2f}".format(
            mean(scores["test_accuracy"]),
            mean(scores["test_precision"]),
            mean(scores["test_recall"]),
            mean(scores["test_f1_score"]),
        )
    )


def view_detailed_metrics(scores):
    """
	This method takes a dictionary outputed by the 'KNNevaluate' method as its input

	It returns the evaluation metric values over all the splits during cross validation based on the data passed to the 'KNNevaluate' method  
	"""
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1_score"]
    for metric in metrics:
        # generates comma seperated string of metric values and omits last comma
        result = "".join(str(val)[:4] + "," for val in scores[metric])[:-1]
        print("All the {} values are {}".format(metric, result))
