import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



def load_split_preprocessed_data(i):

    data = pd.read_csv("../../datasets/winequality.csv")
    Count_Fraud_transacation = len(data["recommend"]==True) 
    fraud_indices= np.array((data.recommend==True).index)
    normal_indices = np.array((data.recommend==False).index)

    undersample_data = undersample(normal_indices,fraud_indices,i,Count_Fraud_transacation,data)

    # since data is skewed, we add epsilon to data and take its logarithm
    undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','alcohol','quality']] = np.add(undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','alcohol','quality']],0.00000001)
    undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']] = np.log(undersample_data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']])
    undersample_data.drop(["quality"],axis=1,inplace=True)
    x_features= undersample_data.drop(["recommend"],axis=1,inplace=False)
    x_labels=undersample_data["recommend"]

    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.3)

    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

#undersampling data
def undersample(normal_indices,fraud_indices,times, Count_Fraud_transacation,data):#times denote the normal data = times*fraud data
    Normal_indices_undersample = np.array(np.random.choice(normal_indices,(times*Count_Fraud_transacation),replace=True))
    undersample_data= np.concatenate([fraud_indices,Normal_indices_undersample])
    undersample_data = data.iloc[undersample_data,:]
    return(undersample_data)
    

