import pandas as pd
import numpy as np

import sklearn
from sklearn import (
    svm,
    tree,
    linear_model,
    neighbors,
    naive_bayes,
    ensemble,
    discriminant_analysis,
    gaussian_process,
)
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import time


def get_algorithms():
    MLA_dict = {
        # Ensemble methods
        "ada": ensemble.AdaBoostClassifier(),
        "bc": ensemble.BaggingClassifier(),
        "etc": ensemble.ExtraTreesClassifier(),
        "gbc": ensemble.GradientBoostingClassifier(),
        "rfc": ensemble.RandomForestClassifier(),
        # Gaussian processes
        "gpc": gaussian_process.GaussianProcessClassifier(),
        # Linear models
        "lr": linear_model.LogisticRegressionCV(),
        "pac": linear_model.PassiveAggressiveClassifier(),
        "rcc": linear_model.RidgeClassifierCV(),
        "sgd": linear_model.SGDClassifier(),
        "per": linear_model.Perceptron(),
        # Navies bayes
        "bnb": naive_bayes.BernoulliNB(),
        "gnb": naive_bayes.GaussianNB(),
        # Nearest neighbour
        "knn": neighbors.KNeighborsClassifier(),
        # SVM
        "svc": svm.SVC(probability=True),
        "nvc": svm.NuSVC(probability=True),
        "lvc": svm.LinearSVC(),
        # Trees
        "dtc": tree.DecisionTreeClassifier(),
        "ets": tree.ExtraTreeClassifier(),
        # Discriminant analysis
        "lda": discriminant_analysis.LinearDiscriminantAnalysis(),
        "qda": discriminant_analysis.QuadraticDiscriminantAnalysis(),
    }
    return MLA_dict


def run_models(MLA, X, y, train_X, train_y, test_X, cv_split):

    # Create table to compare MLA predictions
    MLA_predict = pd.DataFrame()

    # Index through MLA and save performance to table
    MLA_list = []
    for alg in MLA:
        MLA_row = {}
        # Set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_row["MLA Name"] = MLA_name
        MLA_row["MLA Parameters"] = str(alg.get_params())
        # Pipeline for normalisation
        pipe = Pipeline([("scale", StandardScaler()), ("clf", alg)])
        # Score model with cross validation
        cv_results = model_selection.cross_validate(
            pipe, X, y, cv=cv_split, return_train_score=True, n_jobs=-1
        )

        MLA_row["MLA Train Accuracy Mean"] = cv_results["train_score"].mean()
        MLA_row["MLA Test Accuracy Mean"] = cv_results["test_score"].mean()
        # If this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_row["MLA Test Accuracy 3*STD"] = (
            cv_results["test_score"].std() * 3
        )  # Let us know the worst that can happen!
        MLA_row["MLA Time"] = cv_results["fit_time"].mean()

        MLA_list.append(MLA_row)

        # Save MLA predictions that could be used later
        alg.fit(train_X, train_y)
        MLA_predict[MLA_name] = alg.predict(test_X)

    MLA_compare = pd.DataFrame(MLA_list)
    # Print and sort table
    MLA_compare.sort_values(
        by=["MLA Test Accuracy Mean"], ascending=False, inplace=True
    )
    return MLA_compare, MLA_predict


def run_DT_model(X, y, cv_split):
    # base model
    dtree = tree.DecisionTreeClassifier(random_state=42)
    base_results = model_selection.cross_validate(
        dtree, X, y, cv=cv_split, return_train_score=True, n_jobs=-1
    )
    dtree.fit(X, y)

    print("BEFORE DT Parameters: ", dtree.get_params())
    print(
        "BEFORE DT Training accuracy: {:.2f}".format(
            base_results["train_score"].mean() * 100
        )
    )
    print(
        "BEFORE DT Test accuracy: {:.2f}".format(
            base_results["test_score"].mean() * 100
        )
    )
    print(
        "BEFORE DT Test score 3*std: +/- {:.2f}".format(
            base_results["test_score"].std() * 100 * 3
        )
    )
    print(
        "BEFORE DT Test accuracy (min): {:.2f}".format(
            base_results["test_score"].min() * 100
        )
    )
    print("-" * 10)

    # tune hyper-parameters
    param_grid = {
        "criterion": [
            "gini",
            "entropy",
        ],  # scoring methodology; two supported formulas for calculating information gain; default is gini
        "splitter": [
            "best",
            "random",
        ],  # splitting methodology; two supported strategies; default is best
        "max_depth": [
            2,
            4,
            6,
            8,
            10,
            None,
        ],  # max depth that the tree can grow; default is none
        "min_samples_split": [
            2,
            5,
            10,
            0.03,
            0.05,
        ],  # minimum subset size BEFORE new split (fraction is % of total); default is 2
        "min_samples_leaf": [
            1,
            5,
            10,
            0.03,
            0.05,
        ],  # minimum subset size AFTER new split split (fraction is % of total); default is 1
        "max_features": [
            None,
            "auto",
        ],  # max features to consider when performing split; default is None, i.e., all
        "random_state": [
            0
        ],  # seed for random number generator, hence not required to be tuned
    }

    # choose the best model with grid search
    tuned_model = model_selection.GridSearchCV(
        tree.DecisionTreeClassifier(),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv_split,
        return_train_score=True,
        n_jobs=-1,
    )
    tuned_model.fit(X, y)

    print("AFTER DT Parameters: ", tuned_model.best_params_)
    print(
        "AFTER DT Training accuracy: {:.2f}".format(
            tuned_model.cv_results_["mean_train_score"][tuned_model.best_index_] * 100
        )
    )
    print(
        "AFTER DT Test accuracy: {:.2f}".format(
            tuned_model.cv_results_["mean_test_score"][tuned_model.best_index_] * 100
        )
    )
    print(
        "AFTER DT Test 3*std: +/- {:.2f}".format(
            tuned_model.cv_results_["std_test_score"][tuned_model.best_index_] * 100 * 3
        )
    )
    print("-" * 10)
    return tuned_model


def run_voting_model(X, y, cv_split):

    # Using default hyper-parameters
    MLA_dict = get_algorithms()
    # Removing models without attribute 'predict_proba' (required for vote classifier)
    # and models with a 1.0 correlation to another model
    clf_keys = [
        "ada",
        "bc",
        "etc",
        "gbc",
        "rfc",
        "gpc",
        "lr",
        "bnb",
        "gnb",
        "knn",
        "svc",
        "lda",
        "qda",
    ]
    vote_est = []
    for clf_name in clf_keys:
        vote_est.append((clf_name, MLA_dict[clf_name]))

    # Hard Vote or majority rules
    vote_hard = ensemble.VotingClassifier(estimators=vote_est, voting="hard")
    pipe_hard = Pipeline([("scale", StandardScaler()), ("clf", vote_hard)])
    vote_hard_cv = model_selection.cross_validate(
        pipe_hard, X, y, cv=cv_split, return_train_score=True, n_jobs=-1
    )

    print(
        "Hard Voting Training accuracy: {:.2f}".format(
            vote_hard_cv["train_score"].mean() * 100
        )
    )
    print(
        "Hard Voting Test accuracy: {:.2f}".format(
            vote_hard_cv["test_score"].mean() * 100
        )
    )
    print(
        "Hard Voting Test 3*std: +/- {:.2f}".format(
            vote_hard_cv["test_score"].std() * 100 * 3
        )
    )
    print("-" * 10)

    # Soft Vote or weighted probabilities
    vote_soft = ensemble.VotingClassifier(estimators=vote_est, voting="soft")
    pipe_soft = Pipeline([("scale", StandardScaler()), ("clf", vote_soft)])
    vote_soft_cv = model_selection.cross_validate(
        pipe_soft, X, y, cv=cv_split, return_train_score=True, n_jobs=-1
    )

    print(
        "Soft Voting Training accuracy: {:.2f}".format(
            vote_soft_cv["train_score"].mean() * 100
        )
    )
    print(
        "Soft Voting Test accuracy: {:.2f}".format(
            vote_soft_cv["test_score"].mean() * 100
        )
    )
    print(
        "Soft Voting Test 3*std: +/- {:.2f}".format(
            vote_soft_cv["test_score"].std() * 100 * 3
        )
    )
    print("-" * 10)
    return vote_est, vote_hard_cv, vote_soft_cv


def tune_hparams(X, y, cv_split, vote_est):

    # Tune using a suitable set of values
    grid_n_estimator = [10, 50, 100, 300, 1000]
    grid_ratio = [0.1, 0.25, 0.5, 0.75, 1.0]
    grid_learn = [0.01, 0.03, 0.05, 0.1, 0.25]
    grid_max_depth = [2, 3, 4, 6, 8, 10, 12, 14, 16, None]
    grid_min_samples = [5, 10, 0.03, 0.05, 0.10]
    grid_criterion = ["gini", "entropy"]
    grid_bool = [True, False]
    grid_seed = [0]

    # Trying with almost all suitable combinations, but I have commented out
    # some very time-consuming variations (and kept the best one from past runs).
    grid_param = {
        "ada": [
            {
                # AdaBoostClassifier
                # Pipeline requires clf__ to be added before each key
                "clf__n_estimators": grid_n_estimator,  # default: 50
                "clf__learning_rate": grid_learn,  # default: 1
                "clf__algorithm": ["SAMME", "SAMME.R"],  # default: ’SAMME.R'
                "clf__random_state": grid_seed,
            }
        ],
        "bc": [
            {
                # BaggingClassifier
                "clf__n_estimators": grid_n_estimator,  # default: 10
                "clf__max_samples": grid_ratio,  # default: 1.0
                "clf__random_state": grid_seed,
            }
        ],
        "etc": [
            {
                # ExtraTreesClassifier
                "clf__n_estimators": grid_n_estimator,  # default: 10
                "clf__criterion": grid_criterion,  # default: 'gini'
                "clf__max_depth": grid_max_depth,  # default: None
                "clf__random_state": grid_seed,
            }
        ],
        "gbc": [
            {
                # GradientBoostingClassifier
                "clf__loss": ["deviance", "exponential"],  # default: ’deviance’
                "clf__learning_rate": [
                    #                     0.05,
                    0.1,
                ],  # default: 0.1, best: 0.1 (This one takes time)
                "clf__n_estimators": [
                    #                     100,
                    300,
                ],  # default: 100, best: 300 (This one takes time)
                "clf__criterion": [
                    "friedman_mse",
                    "mse",
                    "mae",
                ],  # default: ”friedman_mse”
                "clf__max_depth": grid_max_depth,  # default: 3
                "clf__random_state": grid_seed,
            }
        ],
        "rfc": [
            {
                # RandomForestClassifier
                "clf__n_estimators": grid_n_estimator,  # default: 10
                "clf__criterion": grid_criterion,  # default: 'gini'
                "clf__max_depth": grid_max_depth,  # default: None
                "clf__oob_score": [
                    True,
                    #                     False,
                ],  # default: False, best: True (This one takes time)
                "clf__random_state": grid_seed,
            }
        ],
        "gpc": [
            {
                # GaussianProcessClassifier
                "clf__max_iter_predict": grid_n_estimator,  # default: 100
                "clf__random_state": grid_seed,
            }
        ],
        "lr": [
            {
                # LogisticRegressionCV
                "clf__fit_intercept": grid_bool,  # default: True
                "clf__penalty": ["l1", "l2"],
                "clf__solver": [
                    "newton-cg",
                    "lbfgs",
                    "liblinear",
                    "sag",
                    "saga",
                ],  # default: 'lbfgs'
                "clf__random_state": grid_seed,
            }
        ],
        "bnb": [
            {
                # BernoulliNB
                "clf__alpha": grid_ratio,  # default: 1.0
            }
        ],
        # GaussianNB
        "gnb": [{}],
        "knn": [
            {
                # KNeighborsClassifier
                "clf__n_neighbors": [1, 2, 3, 4, 5, 6, 7],  # default: 5
                "clf__weights": ["uniform", "distance"],  # default: ‘uniform’
                "clf__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            }
        ],
        "svc": [
            {
                # SVC
                "clf__kernel": ["linear", "poly", "rbf", "sigmoid"],
                "clf__C": [1, 2, 3, 4, 5],  # default: 1.0
                "clf__gamma": grid_ratio,  # edfault: 'auto'
                "clf__decision_function_shape": ["ovo", "ovr"],  # default: ovr
                "clf__probability": grid_bool,
                "clf__random_state": grid_seed,
            }
        ],
        # LDA
        "lda": [{}],
        # QDA
        "qda": [{}],
    }

    start_total = time.perf_counter()
    for clf in vote_est:

        # vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm

        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf[1])])

        best_search = model_selection.GridSearchCV(
            estimator=pipe,
            param_grid=grid_param[clf[0]],
            cv=cv_split,
            scoring="accuracy",
            n_jobs=-1,
        )

        start = time.perf_counter()
        best_search.fit(X, y)  # Actual computation happens in this step
        run = time.perf_counter() - start

        best_param = best_search.best_params_
        # Remove 'clf__' from all keys to make it look more readable
        best_param = {
            param_name[5:]: param_val for param_name, param_val in best_param.items()
        }

        print(
            "The best parameter for {} is {} with a runtime of {:.2f} seconds.\n".format(
                clf[1].__class__.__name__, best_param, run
            )
        )

        # Set the best parameters obtained to each classifier
        clf[1].set_params(**best_param)

    run_total = time.perf_counter() - start_total
    print("Total optimization time was {:.2f} minutes.".format(run_total / 60))

    print("-" * 10)
    return grid_param
