#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


def dupli_rows(df):
    print("number of duplicate rows:%d " %(df.duplicated().sum()))
    
def view(df):
    row,col=df.shape
    print("rows:%d columns:%d" %(row,col))

def check_null(df):
    df.info()
    df.isnull().sum() 
    
def frequency(df,x,y):
    data = x.value_counts().sort_index()
    data_df = pd.DataFrame({y: data.index,'Frequency': data.values})
    return(data_df)

def graph(df,p,q,r):
    plt.figure(figsize=(7,5))
    sns.barplot(x = p, y = q, data = df)
    plt.title(r,fontsize=16)
    plt.show()
    
def heat_map(df):
    df1 = df.drop("recommend",axis=1)
    correlationMap = df1.corr()
    plt.figure(figsize=(30,30))
    sns.set(font_scale=3)
    hm = sns.heatmap(correlationMap,cmap = 'Set1', cbar=True, annot=True,vmin=0,vmax =1,center=True, square=True, fmt='.2f', annot_kws={'size': 25},
             yticklabels = df1.columns, xticklabels = df1.columns)
    plt.show()
    
def correlate_binary(df):
    cor = df.corr()
    cor_target = abs(cor["recommend"])
    relevant_features = cor_target[cor_target>0]
    return(relevant_features)
    
def correlate_multi(df):
    df1=df.drop("recommend",axis=1)
    cor = df1.corr()
    cor_target = abs(cor["level"])
    relevant_features = cor_target[cor_target>0]
    return(relevant_features)
    


# In[ ]:




