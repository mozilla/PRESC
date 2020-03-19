from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def Calculate_accuracy(classifier, X_train, y_train ,size):
    scores = []
    split_matrix = pd.DataFrame(columns=[ "KFold" , 'Training_Set', 'Testing_Set', 'Accuracy'])
    test_size = 100 * size
    train_size = 100 - (test_size)
    for i in range(2, 6):
        score = cross_val_score(classifier, X_train, y_train, cv=i , scoring = "accuracy")
        scores.append(score.mean())
        split_matrix = split_matrix.append({'KFold': i ,'Training_Set': train_size, 'Testing_Set': test_size ,
                                            'Accuracy': (score.mean() * 100 )}, ignore_index=True)
    return split_matrix    
  
def visulaize_train_test_split_traversal(split_matrix):
    ax = plt.gca()
    
    print("------------------------------------------------------------------")
    split_matrix.plot(kind='line',x='KFold',y='Accuracy', color='red' ,ax=ax)
    split_matrix.plot(kind='line',x='KFold',y='Testing_Set', color='yellow', ax=ax)
    split_matrix.plot(kind='line',x='KFold',y='Training_Set', color='blue', ax=ax)
    plt.title("Line plot with Accuracy , Training and Testing Set\n")
    plt.ylabel("Accuracy\n")
    plt.xlabel("\nNo of KFolds")
    plt.show()    


def train_test_split_traversal(classifier , vdataset):
    
    X = vdataset.drop(["Class" ,"Class_code"] , axis = 1)
    y = vdataset["Class_code"]
    split_matrix = pd.DataFrame(columns=[ "KFold" , 'Training_Set', 'Testing_Set', 'Accuracy'])
    size = [ 0.2 ,0.4 , 0.6 ,0.8]
    for s in size:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= s, random_state=45)
        matrix  = Calculate_accuracy(classifier, X_train, y_train , s)
        split_matrix = pd.concat([split_matrix, matrix], ignore_index=True)
        visulaize_train_test_split_traversal(matrix)
    return split_matrix        