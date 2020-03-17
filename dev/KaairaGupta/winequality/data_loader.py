import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def load_split_preprocessed_data(i):

    data_path = "../../../datasets/winequality.csv"
    data = pd.read_csv(data_path)
    count_false_recommendation = len(data["recommend"]==True) 
    false_indices= np.array((data.recommend==True).index)
    true_indices = np.array((data.recommend==False).index)

    undersample_data = undersample(true_indices,false_indices,i,count_false_recommendation,data)

    # since data is skewed, we add epsilon to data and take its logarithm
    epsilon = 0.00000001

    undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','alcohol','quality']] = np.add(undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','alcohol','quality']],epsilon)
    undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']] = np.log(undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']])
    undersample_data.drop(["quality"],axis=1,inplace=True)
    x_features= undersample_data.drop(["recommend"],axis=1,inplace=False)
    x_labels=undersample_data["recommend"]
    
    # splitting data on the basis of accuracy with different sets. Accuracy didn't vary much with the split, but gave highest with this number. Hence, you are free to go ahead with any number.

    test_size=0.3
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=test_size)

    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

#undersampling data
def undersample(true_indices,false_indices,times, count_false_recommendation, data):#times denote the true = times*false data
    true_indices_undersample = np.array(np.random.choice(true_indices,(times*count_false_recommendation),replace=True))
    undersample_data= np.concatenate([false_indices,true_indices_undersample])
    undersample_data = data.iloc[undersample_data,:]
    return(undersample_data)
    

