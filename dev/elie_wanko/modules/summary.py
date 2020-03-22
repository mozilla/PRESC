from pandas import DataFrame
from sklearn import metrics
from sklearn.model_selection import train_test_split

import helpers, logreg

def ttsplits(headers=["Test Size", "Accuracy Score", "Precision Score", "Recall Score", "F1 Score"], *, data, test_sizes):
    # Split independent and target features 
    independ_attrs, target_attrs = helpers.independ_target_attr_split(data)

    eval_metrics = []
    for size in test_sizes:
        # Split train and test data subsets
        subsets = train_test_split(independ_attrs, target_attrs, test_size=size, random_state=1)
        
        # Classifier
        lg_pred, lg_true = logreg.classifier(tol=0.000014, subsets=subsets, data=data, solver='liblinear')
        ac = float("{:.4f}".format(metrics.accuracy_score(lg_true, lg_pred)))
        ps = float("{:.4f}".format(metrics.precision_score(lg_true, lg_pred)))
        rs = float("{:.4f}".format(metrics.recall_score(lg_true, lg_pred)))
        f1 = float("{:.4f}".format(metrics.f1_score(lg_true, lg_pred)))

        #Arrary of results
        eval_metrics.append([size, ac, ps, rs, f1])
        
    return DataFrame(eval_metrics, range(len(test_sizes)), headers)
