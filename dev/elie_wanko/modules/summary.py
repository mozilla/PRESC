from numpy import ravel
from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split

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
    independ_attrs = data.filter(data.columns[:-1])
    target_attrs = ravel(
        data.filter(data.columns[-1:])
    )  # Returns a 1-D array, containing the elements of the input

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

    return DataFrame(eval_metrics, range(len(sizes)), headers)
