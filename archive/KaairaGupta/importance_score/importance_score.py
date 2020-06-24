import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def imp_score_singular_datapoints(model, train_features, train_labels, test_features, test_labels, i_start, i_end):
    """
        Inputs: 
            model: model object which must have function model.fit(train_features, train_labels)
            train_features: numpy array of feature values for training
            train_labels: numpy array of correct labels for train_features
            test_features: numpy array of feature values for test data
            test_labels: numpy array of correct labels for test data
            i_start: start index to remove features from train_labels (one-by-one)
            i_end: end index (not inclusive) to remove features from train_labels (one-by-one)
        Output:
            Plot of difference in performance scores (accuracy and f-score) on the test set by removing data points (one-by-one) from training data.
    """
    
    fs_list = []
    acc_list = []
    
    temp_model = model
    temp_model.fit(train_features, train_labels)
    
    pred_full_data = temp_model.predict(test_features)

    fs_full_data = metrics.f1_score(test_labels, pred_full_data)
    acc_full_data = metrics.accuracy_score(test_labels, pred_full_data)
    
    for i in range(i_start,i_end):
        temp_train = np.concatenate((train_features[:i], train_features[i+1:]))
        temp_labels = np.concatenate((train_labels[:i], train_labels[i+1:]))
        temp_model = model
        temp_model.fit(temp_train, temp_labels)
        
        pred = temp_model.predict(test_features)
        
        fs = metrics.f1_score(test_labels, pred)
        acc = metrics.accuracy_score(test_labels, pred)
        
        fs_list.append(fs)
        acc_list.append(acc)

    fig, ax = plt.subplots()
    # line = mlines.Line2D([0, 1], [0, 1], color='black')
    # transform = ax.transAxes
    # line.set_transform(transform)
    # ax.add_line(line)
    fig.suptitle('Importance Score Graph')
    ax.set_xlabel('Removing i\'th data-point')
    ax.set_ylabel('Scores')
    
    plt.plot(-1,fs_full_data,'ro',label="F-Score Full Data") 
    plt.plot(-1,acc_full_data,'ro',label="Accuracy Full Data")
    plt.plot(np.arange(i_end-i_start),fs_list, marker='o', linewidth=1, label='F-Score Values')
    plt.plot(np.arange(i_end-i_start),acc_list, marker='o', linewidth=1, label='Accuracy Values')

    plt.legend()
    plt.show()
