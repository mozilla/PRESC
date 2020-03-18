# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics


def __init__():
    pass


def univ_bar(
    title=None,
    fig_w=10,
    fig_h=5,
    var_names=None,
    hue=None,
    palette=None,
    order=True,
    x_title=None,
    *,
    data,
    column
):
    """Plots a Bar Graph of variable of interest 'column'.
    """
    # Make a deep copy of dataframe
    df_clone = data.copy()
    # Exacts distinct entries of an attricbute(comumn) and Sorts it into a list
    sorted_list = sorted(set(df_clone[column]))

    # Extract and Replace categorical numerical values to Friendgly words(var_names)
    if var_names != None:
        df_clone[column] = df_clone[column].replace(sorted_list, var_names)

    # Plot bar graph
    plt.figure(figsize=(fig_w, fig_h))
    if order == True:
        ax = sns.countplot(
            data=df_clone,
            x=x_title,
            hue=hue,
            palette=palette,
            order=df_clone[column].value_counts().index,
        )
    else:
        ax = sns.countplot(data=df_clone, x=x_title, hue=hue, palette=palette)

    # Adds count (defaulters : non-defaulters) on top of bars
    for p in ax.patches:
        ax.annotate(
            "{:.0f}".format(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 50),
            ha="center",
        )

    # Adds relative percentages(defaulters : non-defaulters) per category on top of bars
    bars = ax.patches
    half = int(len(bars) / 2)
    left_bars = bars[:half]
    right_bars = bars[half:]

    for left, right in zip(left_bars, right_bars):
        height_l = left.get_height()
        height_r = right.get_height()
        total = height_l + height_r

        ax.text(
            left.get_x() + left.get_width() / 2.0,
            height_l + 550,
            "{0:.2%}".format(height_l / total),
            ha="center",
        )
        ax.text(
            right.get_x() + right.get_width() / 2.0,
            height_r + 550,
            "{0:.2%}".format(height_r / total),
            ha="center",
        )

    # Set title
    plt.title(title, fontsize=14)


def univ_hist(title=None, fig_w=18, fig_h=7, _hue=None, bins=10, *, data, column):
    """Plots a Histogram of variable of interest 'column'.
    """
    # Make a deep copy of dataframe
    df_clone = data.copy()

    # Plot hist graph
    plt.figure(figsize=(fig_w, fig_h))
    sns.distplot(df_clone[column], bins=bins)
    # Set title
    plt.title(title, fontsize=14)


def independ_target_attr_split(data):
    """ Splits data into independent and dependent attributes for feautre selection, model training and testing.
    """
    # Split training and testing attributes
    independent_attributes = data.columns[:-1]
    target_attribute = data.columns[-1]

    # Get data indices.
    indices = data.iloc[:]

    return indices[independent_attributes], indices[target_attribute]


def confusion_matrix(cmap="Blues", *, true, pred):
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        metrics.confusion_matrix(true, pred),
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        square=True,
        cmap=cmap,
    )
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    a_score = "Accuracy Score: {:.2f}".format(metrics.accuracy_score(true, pred))
    plt.title(a_score)

    
def highlight_max(data, color='#9dd9a7', ):
    """Highlights the maximum in a Series or DataFrame
    """
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    
