#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:
"""Entropy and Gini index are impurity measures used in decision trees """


def tree_model_gini(X_test, X_train, y_train, d, l):
    """a criterion to minimize the probability of misclassification"""
    clf_gini = DecisionTreeClassifier(
        criterion="gini",
        splitter="random",
        random_state=42,
        max_depth=d,
        min_samples_leaf=l,
    )
    clf_gini.fit(X_train, y_train)
    y_pred = clf_gini.predict(X_test)
    y_prob = clf_gini.predict_proba(X_test)[:, 1]
    return (y_pred, y_prob)


def tree_model_entropy(X_test, X_train, y_train, d, l):
    """a way to measure impurity"""
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=42, max_depth=d, min_samples_leaf=l
    )
    clf_entropy.fit(X_train, y_train)
    y_pred_en = clf_entropy.predict(X_test)
    y_prob_en = clf_entropy.predict_proba(X_test)[:, 1]
    return (y_pred_en, y_prob_en)


def dtree_grid_search(X, y, nfolds):
    """for hyper parameter tuning to choose appropriate parameter values to improve the overall accuracy of our classifier"""
    # create a dictionary of all values we want to test
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 15),
        "min_samples_leaf": range(2, 5),
    }
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    # use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    # fit model to data
    dtree_gscv.fit(X, y)
    return dtree_gscv.best_params_
