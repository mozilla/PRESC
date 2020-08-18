import numpy as np
from math import ceil

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


def explore_cross_validation_kfolds(
    dataset,
    pipeline,
    metrics=["accuracy"],
    kfolds_list=[3, 4, 5, 7, 10],
    repetitions=1,
    minimum_repetitions=False,
):
    """Explore the cross-validation k-fold number with the specified pipeline.

    Choosing the number of k-folds with which to carry out the cross-validation
    is not an obvious decision. This function allows to explore the k-fold space
    to determine the optimal number of k-folds.

    A larger number of k-folds will mean that the training sample for each
    k-fold is closer to the full dataset and that they are more similar among
    them because they will have more overlap, as well as a larger number of
    different scores to average, but the validation subsets will be smaller
    and yield a higher variance. Also, a higher number of k-folds will be more
    computationally demanding.

    For instance, if we divide the data in two k-folds this will correspond to a
    train/test split of 50/50 for each k-fold where we will only obtain two
    different performace values (one per fold) to average. Choosing ten k-folds
    will correspond to a train/test split of 90/10 obtaining ten performance
    values to average, and setting the number of k-folds identical to the number
    of data points corresponds to a leave-one-out cross validation scheme where
    all the data points but one are used for training, and the model is
    validated for one point on each iteration.

    Smaller training samples (i.e., a smaller number of k-folds) make it easier
    that the training subset is not as representative and that the classifier
    has biases, hence, if the whole cross validation process is repeated, there
    will be more variation among those runs. One way to counteract this effect
    for small numbers of k-folds is to run the cross validation several times
    and average all those values.

    Parameters:
        dataset (DataFrame): Pandas dataset.
        pipeline (Pipeline): Can be a scaler and a classifier or only the
            classifier.
        metrics (list of str): So far only "accuracy" is possible, although the
            fit and scoring times are also computed.
        kfolds_list (list): List of the number of k-folds for which to compute
            the cross-validation scores.
        repetitions (int): Number of times to repeat the cross-validation for
            each k-fold number. By default it carries out the cross-validation
            once for each requested number of k-folds.
        minimum_repetitions (bool): If "minimum_repetitions" is set to True,
            "repetitions" becomes the minimum number of repetitions and the
            number of needed cross-validation runs for each k-fold value are
            carried out. For example, the cross validation for five k-folds
            yields five performance values so, if the minimum number of
            repetitions is set to ten, the cross validation will be repeated
            twice.
    Returns:
        kfolds_list (list): List of the number of k-folds for which to compute
            the cross-validation scores.
        scores_summary (dict of list): Dictionary with the different computed
            scores. Each score has a list with the list of average values for
            each k-fold value, and the list of standard deaviations for each
            k-fold value.
    """

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    repetitions_original = repetitions

    scores_summary = None

    for num in kfolds_list:

        # If number of k-folds is smaller than the minimum number of
        # repetitions, repeat the cross-validation for that k-fold number as
        # many times as necessary.
        if minimum_repetitions is True:
            if num <= repetitions_original:
                repetitions = ceil(repetitions_original / num)
            else:
                repetitions = 1

        scores_list = []
        scores_kfold = []

        for repetition in range(repetitions):
            cv = KFold(n_splits=num, shuffle=True)  # , random_state=None)
            scores = cross_validate(
                pipeline, X, y, scoring=metrics, cv=cv, return_estimator=False
            )
            scores_list += [scores]

        # Gather all repetitions of each kfold
        scores_kfold = {metric: [] for metric in scores}

        for element in scores_list:
            for metric in element:
                scores_kfold[metric] += list(element[metric])

        # Consolidate all repetitions
        scores_kfold = {
            metric: [np.mean(scores_kfold[metric]), np.std(scores_kfold[metric])]
            for metric in scores_kfold
        }

        # Gather all kfolds
        if scores_summary is None:
            scores_summary = {metric: [[], []] for metric in scores}

        for metric in scores_kfold:
            scores_summary[metric][0] += [scores_kfold[metric][0]]
            scores_summary[metric][1] += [scores_kfold[metric][1]]

    return kfolds_list, scores_summary
