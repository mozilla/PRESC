#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


def remove_dupli(df):
    df.drop_duplicates(inplace=True)
    return df


def process_binary(df):
    df = df * 1
    return df


def process_multi(df, x):
    df["level"] = x.apply(lambda q: 3 if q <= 4 else 2 if (q > 4 and q <= 6) else 1)
    df.head()


def data_split(x, y, z):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=z, random_state=42
    )
    return (X_train, X_test, y_train, y_test)
