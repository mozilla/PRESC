import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def misclassificationGraphHelper(y_test, preds):
    """
    Helps concat y_test and preds to form a new dataframe to plot misclassifications.
    
    Args: y_test - Actual test label
          preds - Predictions
    Returns: df_miss - A dataframe with 2 columns that defines the misclassified data.
    """
    predsSeries = pd.Series(preds)
    # predsSeries = predsSeries.reindex(index=y_test.index)
    df_miss = pd.concat([y_test, predsSeries], axis=1, join="inner")
    df_miss = df_miss.rename(columns={"defaulted": "y_test", 0: "predsSeries"})
    return df_miss


def misclassificationGraphs(df_miss):
    """
    Plots a graph to show misclassified values.
    
    Agrs: df_miss - The misclassification dataframe obtained from graphReindexHelper
    """
    ax1 = sns.countplot(x="predsSeries", hue="y_test", data=df_miss, palette="Blues")


def plotAOCROC(clf_model_list, clf_model_names, X_test, y_test):
    """
    Plots ROC Curves and calculates AUC.
    
    Args: clf_model - The classification model.
          X_test - Test features
          y_test - Test labels
    """
    plt.figure()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Classifier ROC")
    i = 0
    for clf in clf_model_list:
        predictedprob = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predictedprob[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, lw=2, label=clf_model_names[i] + "ROC area = %0.2f)" % roc_auc
        )
        i = i + 1
    plt.legend(loc="lower right")
    plt.show()
