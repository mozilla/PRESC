from sklearn import metrics
import numpy as np
def model_performance_evaluater_score (model_func,X_train, y_train, X_test, y_test):
    '''returns accuracy of a model when training and testing data and target labels are provided
    inputs:
    model_func : model for fitting training data
    X_train : training data for model fitting
    y_train : training data's target labels
    X_test : testing data for model's accuracy calculation
    y_test: testing data's target labels for model's accuracy calculation'''
    model_func.fit(X_train, y_train)
    # y_pred=model_func.predict(X_test) 
    # return metrics.accuracy_score(y_test, y_pred)
    # I am not sure if accuracy or score is record for senstivity calculation so I commented out accuracy code
    return model_func.score(X_test, y_test) 

def create_parallel_db(X_train, y_train, idx):
    ''' to create a parallel training dataset and corresponding label set
    X_train : training data for model fitting
    y_train : training data's target labels
    idx: index of the datapoint that you want to remove
    return one dataset and one labels set which one less in length than the dataset provided in input of this function'''
    X_parallel_train = np.concatenate((X_train[0:idx], X_train[idx+1:]))
    y_parallel_train = np.concatenate((y_train[0:idx], y_train[idx+1:]))
    return X_parallel_train, y_parallel_train

def create_all_parallel_dbs(X_train, y_train):
    ''' to create all parallel training datasets and corresponding label set, suppose we have a dataset of length m, this will create m parallel dataset each with length m-1
    returns a list of  tuples each with one parallel dataset and corresponding labels set'''
    parallel_dbs = list()
    for i in range(len(X_train)):
        X_parallel_train, y_parallel_train = create_parallel_db(X_train, y_train, i)
        parallel_dbs.append((X_parallel_train, y_parallel_train))
    return parallel_dbs

def calculate_senstivity(model_func, X_train, y_train, X_test, y_test):
    ''' calculates L1 senstivity of dataset and returns maximum senstivity as well as list of all accuracies of parallel datasets'''
    full_dataset_score = model_performance_evaluater_score(model_func,X_train, y_train, X_test, y_test)
    sensitivity = 0
    score_all_db = []
    list_train_data_label =create_all_parallel_dbs(X_train, y_train) #return list containing tuples of training data and label
    #print(len(list_train_data_label ))
    for parallel_train, parallel_label in list_train_data_label:
        parallel_dataset_score =model_performance_evaluater_score(model_func,parallel_train, parallel_label, X_test, y_test)
        dataset_distance = np.abs(full_dataset_score - parallel_dataset_score ) # L1 senstivity
        score_all_db.append(dataset_distance)
        if(dataset_distance > sensitivity):
            sensitivity = dataset_distance
    return sensitivity, score_all_db

def max_senstivity_leakage_datapoint(model_func, X_train, y_train, X_test, y_test):
    ''' this function tells us which datapoint is the most sensitive, it prints its index as well as the datapoint itself'''
    test_check, list_scores = calculate_senstivity(model_func, X_train, y_train, X_test, y_test)
    idx_of_most_sensitive_data = list_scores.index(test_check)
    print("index of most senstive data point in training data",idx_of_most_sensitive_data  )
    print( "Most senstive data point" , X_train[idx_of_most_sensitive_data])

