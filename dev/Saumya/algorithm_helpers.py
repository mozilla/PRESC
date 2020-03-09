import pandas as pd
import numpy as np

import sklearn
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import metrics

import time

def get_algorithms():
    MLA = [
        #Ensemble methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(),

        #Gaussian processes
        gaussian_process.GaussianProcessClassifier(),

        #Linear models
        linear_model.LogisticRegressionCV(),
        linear_model.PassiveAggressiveClassifier(),
        linear_model.RidgeClassifierCV(),
        linear_model.SGDClassifier(),
        linear_model.Perceptron(),

        #Navies bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),

        #Nearest neighbour
        neighbors.KNeighborsClassifier(),

        #SVM
        svm.SVC(probability=True),
        svm.NuSVC(probability=True),
        svm.LinearSVC(),

        #Trees    
        tree.DecisionTreeClassifier(),
        tree.ExtraTreeClassifier(),

        #Discriminant analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis()
        ]
    return MLA

def run_models(MLA, X, y, cv_split):
    
    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    MLA_predict = pd.DataFrame()

    # index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        #score model with cross validation
        cv_results = model_selection.cross_validate(alg, X, y, cv  = cv_split, return_train_score=True, n_jobs=-1)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
        #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let us know the worst that can happen!


        #save MLA predictions that could be used later
        alg.fit(X, y)
        MLA_predict[MLA_name] = alg.predict(X)

        row_index += 1


    #print and sort table
    MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    return MLA_compare, MLA_predict

def run_DT_model(X, y, cv_split):
    #base model
    dtree = tree.DecisionTreeClassifier(random_state = 42)
    base_results = model_selection.cross_validate(dtree, X, y, cv  = cv_split, return_train_score=True, n_jobs=-1)
    dtree.fit(X, y)

    print('BEFORE DT Parameters: ', dtree.get_params())
    print("BEFORE DT Training accuracy: {:.2f}". format(base_results['train_score'].mean()*100)) 
    print("BEFORE DT Test accuracy: {:.2f}". format(base_results['test_score'].mean()*100))
    print("BEFORE DT Test score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))
    print("BEFORE DT Test accuracy (min): {:.2f}". format(base_results['test_score'].min()*100))
    print('-'*10)


    #tune hyper-parameters
    param_grid = {'criterion': ['gini', 'entropy'],  #scoring methodology; two supported formulas for calculating information gain; default is gini
                  'splitter': ['best', 'random'], #splitting methodology; two supported strategies; default is best
                  'max_depth': [2,4,6,8,10,None], #max depth that the tree can grow; default is none
                  'min_samples_split': [2,5,10,.03,.05], #minimum subset size BEFORE new split (fraction is % of total); default is 2
                  'min_samples_leaf': [1,5,10,.03,.05], #minimum subset size AFTER new split split (fraction is % of total); default is 1
                  'max_features': [None, 'auto'], #max features to consider when performing split; default is None, i.e., all
                  'random_state': [0] #seed for random number generator, hence not required to be tuned
                 }

    #choose the best model with grid search
    tuned_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True, n_jobs=-1)
    tuned_model.fit(X, y)

    print('AFTER DT Parameters: ', tuned_model.best_params_)
    print("AFTER DT Training accuracy: {:.2f}". format(tuned_model.cv_results_['mean_train_score'][tuned_model.best_index_]*100)) 
    print("AFTER DT Test accuracy: {:.2f}". format(tuned_model.cv_results_['mean_test_score'][tuned_model.best_index_]*100))
    print("AFTER DT Test 3*std: +/- {:.2f}". format(tuned_model.cv_results_['std_test_score'][tuned_model.best_index_]*100*3))
    print('-'*10)
    return tuned_model

def run_voting_model(X, y, cv_split):
    
    #Removed models without attribute 'predict_proba' (required for vote classifier) and models with a 1.0 correlation to another model
    #Using default hyper-parameters
    vote_est = [
        #Ensemble Methods
        ('ada', ensemble.AdaBoostClassifier()),
        ('bc', ensemble.BaggingClassifier()),
        ('etc',ensemble.ExtraTreesClassifier()),
        ('gbc', ensemble.GradientBoostingClassifier()),
        ('rfc', ensemble.RandomForestClassifier()),

        #Gaussian Processes
        ('gpc', gaussian_process.GaussianProcessClassifier()),

        #Linear Models
        ('lr', linear_model.LogisticRegressionCV()),

        #Navies Bayes
        ('bnb', naive_bayes.BernoulliNB()),
        ('gnb', naive_bayes.GaussianNB()),

        #Nearest Neighbor
        ('knn', neighbors.KNeighborsClassifier()),

        #SVM
        ('svc', svm.SVC(probability=True)),

        #Discriminant analysis
        ('lda', discriminant_analysis.LinearDiscriminantAnalysis()),
        ('qda', discriminant_analysis.QuadraticDiscriminantAnalysis())


    ]


    #Hard Vote or majority rules
    vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
    vote_hard_cv = model_selection.cross_validate(vote_hard, X, y, cv  = cv_split, return_train_score=True, n_jobs=-1)
    vote_hard.fit(X, y)

    print("Hard Voting Training accuracy: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
    print("Hard Voting Test accuracy: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
    print("Hard Voting Test 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
    print('-'*10)


    #Soft Vote or weighted probabilities
    vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
    vote_soft_cv = model_selection.cross_validate(vote_soft, X, y, cv  = cv_split, return_train_score=True, n_jobs=-1)
    vote_soft.fit(X, y)

    print("Soft Voting Training accuracy: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
    print("Soft Voting Test accuracy: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
    print("Soft Voting Test 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
    print('-'*10)
    return vote_est, vote_hard_cv, vote_soft_cv

def tune_hparams(X, y, cv_split, vote_est):
    
    #Tune using a suitable set of values
    grid_n_estimator = [10, 50, 100, 300, 1000]
    grid_ratio = [.1, .25, .5, .75, 1.0]
    grid_learn = [.01, .03, .05, .1, .25]
    grid_max_depth = [2, 3, 4, 6, 8, 10, 12, 14, 16, None]
    grid_min_samples = [5, 10, .03, .05, .10]
    grid_criterion = ['gini', 'entropy']
    grid_bool = [True, False]
    grid_seed = [0]

    #Trying with almost all suitable combinations
    grid_param = [
                [{
                #AdaBoostClassifier
                'n_estimators': grid_n_estimator, #default: 50
                'learning_rate': grid_learn, #default: 1
                'algorithm': ['SAMME', 'SAMME.R'], #default: ’SAMME.R'
                'random_state': grid_seed
                }],


                [{
                #BaggingClassifier
                'n_estimators': grid_n_estimator, #default: 10
                'max_samples': grid_ratio, #default: 1.0
                'random_state': grid_seed
                 }],


                [{
                #ExtraTreesClassifier
                'n_estimators': grid_n_estimator, #default: 10
                'criterion': grid_criterion, #default: 'gini'
                'max_depth': grid_max_depth, #default: None
                'random_state': grid_seed
                 }],


                [{
                #GradientBoostingClassifier
                'loss': ['deviance', 'exponential'], #default: ’deviance’
                'learning_rate': [0.05, 0.1], #default: 0.1, best: 0.1 (This one takes time)
                'n_estimators': [100, 300], #default: 100, best: 300 (This one takes time)
                'criterion': ['friedman_mse', 'mse', 'mae'], #default: ”friedman_mse”
                'max_depth': grid_max_depth, #default: 3   
                'random_state': grid_seed
                 }],


                [{
                #RandomForestClassifier
                'n_estimators': grid_n_estimator, #default: 10
                'criterion': grid_criterion, #default: 'gini'
                'max_depth': grid_max_depth, #default: None
                'oob_score': [True, False], #default: False, best: True (This one takes time)
                'random_state': grid_seed
                 }],

                [{    
                #GaussianProcessClassifier
                'max_iter_predict': grid_n_estimator, #default: 100
                'random_state': grid_seed
                }],


                [{
                #LogisticRegressionCV
                'fit_intercept': grid_bool, #default: True
                'penalty': ['l1','l2'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: 'lbfgs'
                'random_state': grid_seed
                 }],


                [{
                #BernoulliNB
                'alpha': grid_ratio, #default: 1.0
                 }],


                #GaussianNB
                [{}],

                [{
                #KNeighborsClassifier
                'n_neighbors': [1,2,3,4,5,6,7], #default: 5
                'weights': ['uniform', 'distance'], #default: ‘uniform’
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }],


                [{
                #SVC
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [1,2,3,4,5], #default: 1.0
                'gamma': grid_ratio, #edfault: 'auto'
                'decision_function_shape': ['ovo', 'ovr'], #default: ovr
                'probability': grid_bool,
                'random_state': grid_seed
                 }],

                #LDA
                [{}],

                #QDA
                [{}]

            ]



    start_total = time.perf_counter()
    for clf, param in zip (vote_est, grid_param):

        #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm    

        start = time.perf_counter()        
        best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'accuracy', n_jobs=-1)
        best_search.fit(X, y)
        run = time.perf_counter() - start

        best_param = best_search.best_params_
        print('The best parameter for {} is {} with a runtime of {:.2f} seconds.\n'.format(clf[1].__class__.__name__, best_param, run))
        #Set the best parameters obtained to each classifier
        clf[1].set_params(**best_param) 


    run_total = time.perf_counter() - start_total
    print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

    print('-'*10)
    return grid_param



