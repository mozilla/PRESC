import numpy as np
from math import ceil

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


def explore_cross_validation_kfolds(
    dataset,
    pipeline,
    metrics=["accuracy"],
    kfolds_list=[3, 4, 5, 7, 10],
    repetitions=10,
    minimum_kfolds=False,
):
    """
    Explore the cross-validation k-fold number with the specified pipeline.

    This function allows to explore the k-fold space to determine the optimal
    number of k-folds for a particular problem. An arbitrary list of different
    k-fold values can be specified, and the cross validation can be repeated the
    same number of times for each k-fold value or, alternatively, adjusted for
    each value so that a minimum count of folds will have been computed.

    Parameters
    ----------
    dataset : DataFrame
        Pandas dataset.
    pipeline : Pipeline
        Can be a scaler and a classifier or only the classifier.
    metrics : list of str
        So far only "accuracy" is possible, although the fit and scoring times are also computed.
    kfolds_list : list
        List of the k-fold values for which to compute the cross-validation scores.
    repetitions : int
        - Number of times to repeat the cross-validation for each k-fold number (if "minimum_kfolds" is set to False)
        - Or minimum number of k-folds that must be computed (if "minimum_kfolds" is set to True). In the latter case, the "repetitions" value must be larger than the largest k-folds value. By default it carries out the cross-validation ten times for each requested k-fold value.
    minimum_kfolds : bool
        If "minimum_kfolds" is set to True, "repetitions" becomes the minimum number of k-folds.
        The number of needed cross-validation runs for each k-fold value are computed so that
        the total number of k-folds after all computations reach that minimum. For example,
        the cross validation for five k-folds yields five performance values so, if the
        minimum number of repetitions is set to ten, the cross validation will be repeated twice.

    Returns
    -------
    kfolds_list :list
        List of the number of k-folds for which to compute the cross-validation scores.
    scores_summary : dict of list
        Dictionary with the different computed scores. Each score has a list with the list
        of average values for each k-fold value, and the list of standard deviations for each
        k-fold value. If "repetitions" is set to one, the standard deviation values correspond
        to those computed for each individual cross-validation.
    """

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    repetitions_original = repetitions
    if minimum_kfolds is True and repetitions <= max(kfolds_list):
        print(
            "WARNING: 'minimum_kfolds' is set to True but the total number "
            "of k-folds (set through \nthe parameter 'repetitions') is smaller "
            "or equal than the largest k-fold value."
        )

    scores_summary = None

    for num in kfolds_list:

        # If the number of k-folds for a single cross-validation run is smaller
        # than the minimum number of repetitions, repeat the cross-validation
        # for that k-fold value as many times as necessary.
        if minimum_kfolds is True:
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
                if len(scores_list) == 1:
                    scores_kfold[metric] += list(element[metric])
                else:
                    scores_kfold[metric] += [np.mean(element[metric])]

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
