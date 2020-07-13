#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


def dupli_rows(df):
    print("number of duplicate rows:%d " % (df.duplicated().sum()))


def view(df):
    row, col = df.shape
    print("rows:%d columns:%d" % (row, col))


def check_null(df):
    df.info()
    df.isnull().sum()


def frequency(df, x, y):
    data = x.value_counts().sort_index()
    data_df = pd.DataFrame({y: data.index, "Frequency": data.values})
    return data_df


def graph(df, p, q, r):
    plt.figure(figsize=(7, 5))
    sns.barplot(x=p, y=q, data=df)
    plt.title(r, fontsize=16)
    plt.show()


def correlate_plot(df):
    df1 = df.drop("recommend", axis=1)
    cor = df1.corr()
    cor_target = abs(cor["quality"])
    relevant_features = cor_target[cor_target > 0]
    return relevant_features


# In[ ]:
