#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


def tree_model_gini(X_test,X_train,y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini",  splitter="random",random_state = 42,max_depth=5, min_samples_leaf=2)
    clf_gini.fit(X_train, y_train)
    y_pred = clf_gini.predict(X_test)
    y_prob = clf_gini.predict_proba(X_test)[:,1]
    return (y_pred,y_prob)

def tree_model_entropy(X_test,X_train,y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 42,max_depth=5, min_samples_leaf=2)
    clf_entropy.fit(X_train, y_train)
    y_pred_en = clf_entropy.predict(X_test)
    y_prob_en = clf_entropy.predict_proba(X_test)[:,1]
    return(y_pred_en,y_prob_en)

