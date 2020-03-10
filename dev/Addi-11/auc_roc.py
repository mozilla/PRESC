import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from itertools import cycle
from scipy.interpolate import interpn
from sklearn.preprocessing import label_binarize

def plot_roc_curve(fper, tper, roc_auc, n_classes):  
    
    # Aggregating all false positive rates
    all_fper = np.unique(np.concatenate([fper[i] for i in range(n_classes)]))
    lw = 2

    # Plot all ROC curves
    plt.figure()
    plt.plot(fper["micro"], tper["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)


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

    # y_test = label_binarize(y_test, classes=n_classes)
    # if len(n_classes) == 2:
        # y_test = np.hstack((1 - y_test, y_test))

    for i in range(2):
        print(i)
        fper[i], tper[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fper[i], tper[i])

    # Compute micro-average ROC curve and ROC area
    fper["micro"], tper["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fper["micro"], tper["micro"])

    plot_roc_curve(fper, tper, roc_auc, n_classes)


