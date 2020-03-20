import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def graphReindexHelper(y_test, preds):
    """
    Helps with reindexing the preds series to the indices for y_test and form misclassified dataframe.
    Args: y_test - Reference series for reindexing.
          preds - To be reindexed acc to y_test
    Returns: df_miss - A dataframe with 2 columns that defines the misclassified data.
    """
    predsSeries = pd.Series(preds)
    predsSeries.reindex(index=y_test.index)
    df_miss = pd.concat([y_test, predsSeries], axis=1, join="inner")
    df_miss = df_miss.rename(columns={"defaulted": "y_test", 0: "predsSeries"})
    return df_miss


def misclassificationGraphs(df_miss):
    """
    Plots a graph to show default values
    Agrs: df_miss - The misclassification dataframe obtained from graphReindexHelper
    """
    ax1 = sns.countplot(x="predsSeries", hue="y_test", data=df_miss, palette="Blues")
