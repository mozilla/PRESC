#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# This function takes in the test set, the target and the prediction and returns the total length of the test set and how many datapoints were missclassified

# In[ ]:


def misclassified_data(df, target, prediction):
    misclassified = df[df[target] != df[prediction]]
    print('Total Test Set: {}'.format(len(df)))
    print("Number of Misclassified Datapoints: {}".format(len(misclassified)))


# In[ ]:




