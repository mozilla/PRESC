import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer

def plot_roc_curve(fper, tper, roc_auc, n_classes):  
    
    # Aggregating all false positive rates
    all_fper = np.unique(np.concatenate([fper[i] for i in range(n_classes)]))
    lw = 2

    # Then interpolate all ROC curves at this points
    mean_tper = np.zeros_like(all_fper)
    for i in range(n_classes):
        mean_tper += np.interp(all_fper, fper[i], tper[i])

    # Finally average it and compute AUC
    mean_tper /= n_classes


    fper["macro"] = all_fper
    tper["macro"] = mean_tper
    roc_auc["macro"] = auc(fper["macro"], tper["macro"])




    # Plot all ROC curves
    plt.figure()
    plt.plot(fper["micro"], tper["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fper["macro"], tper["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fper[i], tper[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def auc_roc(y_test, y_score):
    # y_test = np.array(y_test)
    # y_score = np.array(y_score)
    fper = dict()
    tper = dict()
    roc_auc = dict()
    n_classes = 2
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)
    y_score = lb.fit_transform(y_score)

    if n_classes == 2:
        y_test = np.hstack((1 - y_test, y_test))
        y_score = np.hstack((1 - y_score, y_score))

    # print("y test : ",y_test, "s",y_test.shape)
    # print("y score : ",y_score, "s",y_score.shape)

    for i in range(n_classes):
        fper[i], tper[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fper[i], tper[i])    


    # Compute micro-average ROC curve and ROC area
    fper["micro"], tper["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fper["micro"], tper["micro"])

    plot_roc_curve(fper, tper, roc_auc, n_classes)


