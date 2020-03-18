# Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def univ_bar(
    title=None,
    fig_w=10,
    fig_h=5,
    var_names=None,
    hue=None,
    palette=None,
    order=False,
    x_title=None,
    *,
    data,
    column
):
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
        sns.countplot(
            data=df_clone,
            x=x_title,
            hue=hue,
            palette=palette,
            order=df_clone[column].value_counts().index,
        )
    else:
        sns.countplot(data=df_clone, x=x_title, hue=hue, palette=palette)
    # Set title
    plt.title(title, fontsize=14)


def univ_hist(title=None, fig_w=18, fig_h=7, _hue=None, bins=10, *, data, column):
    # Make a deep copy of dataframe
    df_clone = data.copy()

    # Plot hist graph
    plt.figure(figsize=(fig_w, fig_h))
    sns.distplot(df_clone[column], bins=bins)
    # Set title
    plt.title(title, fontsize=14)
