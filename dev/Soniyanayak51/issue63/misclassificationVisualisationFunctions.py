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
