import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def load_split_preprocessed_data():
    data = pd.read_csv("../../datasets/winequality.csv")
    
    # since data is skewed, we add epsilon to data and take its logarithm
    data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','alcohol','quality']] = np.add(data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','alcohol','quality']],0.00000001)
    data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']] = np.log(data[['fixed acidity','volatile acidity','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']])

    sc = StandardScaler()
    train = data.drop('recommend',axis = 1)
    train = train.drop('quality', axis =1)
    X_train, X_test, y_train, y_test = train_test_split(train, data.recommend, test_size=0.3,random_state=109)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    lda = LDA()
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    return X_train, X_test, y_train, y_test
    

