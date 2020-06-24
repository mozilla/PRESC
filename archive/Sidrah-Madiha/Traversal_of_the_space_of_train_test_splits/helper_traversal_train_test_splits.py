import pandas as pd
import numpy as np
import helper_performance_evaluater_over_folds as single_eval
from allcustommodules import train_data_test_data_split

# from sklearn.model_selection import  cross_val_score, KFold
import matplotlib.pyplot as plt


def performance_evaluater_over_train_test_splits(classifier, X, y, no_of_cv=5):
    """returns a table with columns containing % of Train Data, % of Test Data, No. of folds used, Average metric Score across folds"""
    test_sizes = np.arange(0.01, 1.00, 0.05)
    all_percent_metric_dfs = []
    for each_size in test_sizes:
        X_train, X_test, y_train, y_test = train_data_test_data_split(
            X, y, test_size=each_size
        )
        train_data_percent_list = [int((1.0 - each_size) * 100)] * (no_of_cv - 1)
        test_data_percent_list = [int((each_size) * 100)] * (no_of_cv - 1)
        percent_list_table = pd.DataFrame(
            {
                "% of Training Data": train_data_percent_list,
                "% of Testing Data": test_data_percent_list,
            }
        )
        folds_metric_table = single_eval.performance_evaluater_over_folds(
            classifier, no_of_cv, X_train, y_train
        )  # this returns an indexes columm which I need to remove before concat
        #        folds_metric_table.reset_index()
        unit_perecent_folds_metric = pd.concat(
            [percent_list_table, folds_metric_table], axis=1
        )
        all_percent_metric_dfs.append(unit_perecent_folds_metric)

    performance_evaluatar = pd.concat(all_percent_metric_dfs, axis=0).reset_index(
        drop=True
    )
    return performance_evaluatar


def visualising_performance_evaluater_over_splits(CV_data_split):
    """ returns series of graphs, one for each split ratio, showing avg metric score across folds"""

    # print(len(CV_data_split.groupby(['% of Training Data', '% of Testing Data'])))
    for key, group in CV_data_split.groupby(
        ["% of Training Data", "% of Testing Data"]
    ):

        print(
            "\t========== For "
            + str(key[1])
            + "% train / "
            + str(key[0])
            + "% test ==========="
        )
        table = group.loc[:, ["No. of folds", "Average metric Score"]]
        # print(g[['No. of folds ','Average metric Score'] ])
        single_eval.visualising_performance_evaluater_over_folds(table)
