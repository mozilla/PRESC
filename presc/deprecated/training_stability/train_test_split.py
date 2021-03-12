from datetime import datetime

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def explore_test_split_ratio(
    dataset,
    classifier,
    scaler=None,
    metric="accuracy",
    num_test_fractions=10,
    random_tries=10,
):
    """
    Explore test-train split ratios with the specified scaler and model.

    This function trains the specified model with the given dataset,
    rescaling the variables and compensating during training with
    weights for any imbalance of the sample. It systematically trains
    the model dividing the data into the training and test subsets in
    different proportions.

    Parameters
    ----------
    dataset : DataFrame
        Pandas dataset.
    classifier : classifier
        Any sci-kit learn classifier.
    scaler : scaler
        Any sci-kit learn scaler. The default value is not using any scaler at all.
    metric : str
        Choose among any of the following classification metrics for binary
            - "accuracy" (default)
            - "balanced_accuracy"
            - "average_precision"
            - "f1"
            - "precision"
            - "recall"
            - "true_positives"
            - "false_positives"
            - "true_negatives"
            - "false_negatives"
            - "true_positives_fraction"
            - "false_positives_fraction"
            - "true_negatives_fraction"
            - "false_negatives_fraction"

            For multiclass problems only "accuracy" and "balance accuracy" are
            available.
    num_test_fractions : int
        Number of different test data fractions to explore (fractions between 0 and 1). Default is 10.
    random_tries : int
        Number of randomised trainings to carry out for each test data fraction. Default is 10.

    Returns
    -------
    test_sizes : numpy array
        List of explored test data fractions.
    averages : numpy array
        Average score of the randomised trainings for each test fraction.
    standard_deviations : numpy array
        Standard deviation of the score of the randomised trainings for each test fraction.
    """
    # Load sample dataset
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Generate a list with the requested number of test data fractions
    test_sizes = np.linspace(
        1.0 / (num_test_fractions + 1),
        1.0,
        num=num_test_fractions,
        endpoint=False,
    )

    counter = 0
    score = []
    for fraction in test_sizes:
        score_random = []
        for random_number in range(random_tries):

            # Roughly estimate duration of calculation
            if counter == 0:
                start = datetime.now()
            elif counter == 1:
                interval1 = datetime.now() - start
                total_time = interval1 * num_test_fractions * random_tries / 1.5
                iterations = num_test_fractions * random_tries - counter
                print(
                    f"{interval1} (hh:mm:ss.ss) for {counter} iteration/s. \n"
                    f"There are {iterations} iterations left. \n"
                    f"Estimated total running time: {total_time} (hh:mm:ss.ss)"
                )
            counter += 1

            # Split data into test and train subsets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=fraction, random_state=random_number
            )

            # Rescale all variables if a scaler has been defined
            if scaler is not None:
                scaler = scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Train model
            classifier.fit(X_train_scaled, y_train)

            # Gather predicted classes
            y_predicted = classifier.predict(X_test_scaled)

            # Here we choose and compute the metric
            if metric == "accuracy":
                score_random += [accuracy_score(y_test, y_predicted)]
            if metric == "balanced_accuracy":
                score_random += [balanced_accuracy_score(y_test, y_predicted)]
            if metric == "average_precision":
                score_random += [average_precision_score(y_test, y_predicted)]
            if metric == "f1":
                score_random += [f1_score(y_test, y_predicted)]
            if metric == "precision":
                score_random += [precision_score(y_test, y_predicted)]
            if metric == "recall":
                score_random += [recall_score(y_test, y_predicted)]
            if metric == "true_negatives":
                score_random += [confusion_matrix(y_test, y_predicted)[0, 0]]
            if metric == "false_positives":
                score_random += [confusion_matrix(y_test, y_predicted)[0, 1]]
            if metric == "false_negatives":
                score_random += [confusion_matrix(y_test, y_predicted)[1, 0]]
            if metric == "true_positives":
                score_random += [confusion_matrix(y_test, y_predicted)[1, 1]]
            if metric == "true_negatives_fraction":
                score_random += [
                    confusion_matrix(y_test, y_predicted)[0, 0] / len(y_predicted)
                ]
            if metric == "false_positives_fraction":
                score_random += [
                    confusion_matrix(y_test, y_predicted)[0, 1] / len(y_predicted)
                ]
            if metric == "false_negatives_fraction":
                score_random += [
                    confusion_matrix(y_test, y_predicted)[1, 0] / len(y_predicted)
                ]
            if metric == "true_positives_fraction":
                score_random += [
                    confusion_matrix(y_test, y_predicted)[1, 1] / len(y_predicted)
                ]

        score += [score_random]

    # Gather average scores and standard deviations of all fractions
    score_summary = []
    for scores_list in score:
        score_summary += [(np.mean(scores_list), np.std(scores_list))]

    averages = np.array([score_summary[x][0] for x in range(len(score_summary))])
    standard_deviations = np.array(
        [score_summary[x][1] for x in range(len(score_summary))]
    )

    real_total_time = datetime.now() - start
    print(f"Real total running time: {real_total_time} (hh:mm:ss.ss)")

    print(
        "\nIndex of point where " + metric + " has the smallest standard "
        f"deviation: {standard_deviations.argmin()}"
    )
    print(
        "\nTest fraction where " + metric + " has smallest standard "
        f"deviation: {test_sizes[standard_deviations.argmin()]:.4f}"
        "\nAverage " + metric + " at test fraction with the smallest standard "
        f"deviation: {averages[standard_deviations.argmin()]:.4f}"
    )

    return test_sizes, averages, standard_deviations
