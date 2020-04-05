import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def tune(data_train, target_train):
    """This function performs cross validation in order to find the hyperparameter value that will 
    lead to maximum model performance. """
    param_grid = [
        {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
        {"C": [1, 10, 100, 1000], "gamma": [0.001, 0.0001], "kernel": ["rbf"]},
    ]
    """The parameter grid takes into consideration both the linear and rbf kernels. 
    This gives statistical backing to the final choice of a kernel to use. """
    svc = SVC()
    # The verbose parameter can be added to GridSearchCV so that its progress can be visible.
    optimized_svc = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, iid=True)
    optimized_svc.fit(data_train, target_train)
    params = optimized_svc.best_params_
    # score = optimized_svc.best_score_
    return params


def s_vee_c(data_train, data_test, target_train, target_test):
    """This function trains and tests a support vector classifier using tuned hyperparameters gotten
    from the tune function. Testing and evaluation of the model is carried out on a held out part of 
    the dataset.  """
    params = tune(data_train, target_train)
    svc = SVC(**params)
    svc.fit(data_train, target_train)
    # use fitted model to predict
    target_pred_svc = svc.predict(data_test)
    acc = accuracy_score(target_test, target_pred_svc)
    return (acc, target_pred_svc)
