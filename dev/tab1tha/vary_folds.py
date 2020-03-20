from sklearn.model_selection import cross_val_score
import numpy as np
import time

def vary_folds(estimator, data, target):
    folds = []
    avg_score = []
    duration = []
    #for 1000: ValueError: n_splits=219 cannot be greater than the number of members in each class.
    #The number of folds ought to be positive and greater than 1. Hence the starting point of range() is two
    for fold in range(2,51):
        start = time.time()
        scores = cross_val_score(estimator, data, target, cv = fold)
        mean_score = np.mean(scores)
        stop = time.time()
        #update lists
        avg_score.append(mean_score)
        folds.append(fold)
        duration.append(stop-start)
    return (folds, avg_score, duration)
