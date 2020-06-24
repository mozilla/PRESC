from numpy import ravel, mean
from pandas import DataFrame
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

import helpers, logreg


def ttsplits(
    headers=[
        "Test Size",
        "Accuracy Score",
        "Precision Score",
        "Recall Score",
        "F1 Score",
    ],
    *,
    data,
    sizes
):
    """Builds a table of distribution of scores at different train:test splits
    """

    # Split independent and target features
    independ_attrs = df_data.filter(df_data.columns[:-1])
    target_attrs = df_data.filter(df_data.columns[-1:])

    eval_metrics = []
    for s in sizes:
        # Split train and test data subsets
        subsets = train_test_split(
            independ_attrs, target_attrs, test_size=s, random_state=1
        )
        # Classifier
        lg_pred, lg_true = logreg.classifier(
            tol=0.000014, subsets=subsets, data=data, solver="liblinear"
        )

        ac = float("{:.4f}".format(metrics.accuracy_score(lg_true, lg_pred)))
        ps = float("{:.4f}".format(metrics.precision_score(lg_true, lg_pred)))
        rs = float("{:.4f}".format(metrics.recall_score(lg_true, lg_pred)))
        f1 = float("{:.4f}".format(metrics.f1_score(lg_true, lg_pred)))

        # Arrary of results
        eval_metrics.append([s, ac, ps, rs, f1])

    return DataFrame(eval_metrics, range(len(s)), headers)


def cv_folds(
    headers=[
        "K Folds",
        "Accuracy Score",
        "Precision Score",
        "Recall Score",
        "F1 Score",
    ],
    *,
    data,
    sizes
):
    """Builds a table of distribution of scores at different kfolds splits
    """

    # Split train and test data subsets
    independ_attrs = data.filter(data.columns[:-1])
    target_attrs = ravel(
        data.filter(data.columns[-1:])
    )  # Return a 1-D array, containing the elements of the input
    # Classifier and Scores
    lg = LogisticRegression(tol=0.000014, solver="liblinear")

    eval_metrics = []
    for s in sizes:
        ac = mean(
            cross_val_score(lg, independ_attrs, target_attrs, cv=s, scoring="accuracy")
        )
        ps = mean(
            cross_val_score(lg, independ_attrs, target_attrs, cv=s, scoring="precision")
        )
        rs = mean(
            cross_val_score(lg, independ_attrs, target_attrs, cv=s, scoring="recall")
        )
        f1 = mean(cross_val_score(lg, independ_attrs, target_attrs, cv=s, scoring="f1"))

        # Arrary of results
        eval_metrics.append([s, ac, ps, rs, f1])

    return DataFrame(eval_metrics, range(len(sizes)), headers)
