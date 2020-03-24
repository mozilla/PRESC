import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib as plt
import seaborn as sn

def get_features(file_path):
# Given a file path , return feature matrix and target labels 
    data = pd.read_csv(file_path)
    y = data['Class'].values
    for i in range(0,len(y)):
        if(y[i]=='opel'):
            y[i]=0
        elif(y[i]=='saab'):
            y[i]=1
        elif(y[i]=='bus'):
            y[i]=2
        elif(y[i]=='van'):
            y[i]=3
    y = y.astype('int')
    phi = data.values
    phi = phi[:,0:len(phi[0])-1]

    return phi, y

def split_data(phi, y, sets):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    count = int(len(y)/sets);
    for i in range(0,sets):
       
        test_x.append(phi[i*count:(i+1)*count])
        test_y.append(y[i*count:(i+1)*count])
        
        train_x.append(np.delete(phi,np.s_[i*count:(i+1)*count],0))
        train_y.append(np.delete(y,np.s_[i*count:(i+1)*count],0))
    if(len(y)%sets!=0):    
        test_x.append(phi[sets*count:])
        test_y.append(y[sets*count:])
        train_x.append(np.delete(phi,np.s_[sets*count:],0))
        train_y.append(np.delete(y,np.s_[sets*count:],0))
    
    return(train_x,train_y,test_x,test_y)

def evaluate(est, label):
    
    error = (est-label)
    correct = np.count_nonzero(error ==0)
    accuracy = float(correct)/len(est)
    
    return accuracy

def train_perceptron(x,y):
    sets = len(x)
    model = []
    for i in range(0,sets):
        clf = Perceptron(penalty=None, alpha=0.001, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, 
                        verbose=0, eta0=1.0, n_jobs=None, random_state=0, early_stopping=True, 
                        validation_fraction=0.1, n_iter_no_change=100, class_weight=None, warm_start=True)
        model.append(clf.fit(x[i],y[i]))
        
    return model

def train_decTree(x,y):
    sets = len(x)
    model = []
    for i in range(0,sets):
        clf = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=2, 
                             min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features=None, 
                             random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                             class_weight=None)
        model.append(clf.fit(x[i],y[i]))
        
    return model


def train_neuralNet(x,y):
    sets = len(x)
    model = []
    for i in range(0,sets):
        clf = MLPClassifier(hidden_layer_sizes=(80, ), activation='logistic', solver='adam', alpha=0.0001, 
            batch_size=500, learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, 
            max_iter=1000, shuffle=True, random_state=None, tol=0.001, verbose=False, warm_start=True, 
            momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
            beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
        model.append(clf.fit(x[i],y[i]))
        
    return model


def train_svm(x,y):
    sets = len(x)
    model = []
    for i in range(0,sets):
        clf = SVC(C=10000.0, kernel='rbf', degree=10, gamma='scale', coef0=0.0, shrinking=True, 
            probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
            decision_function_shape='ovr', random_state=None)
        model.append(clf.fit(x[i],y[i]))
        
    return model

        
def scores(model,x,y):
    accuracy = []
    conf_mat=[]
    # auc_score = []
    for i in range(0,len(model)):
        estimated = model[i].predict(x[i])
        # for j in range(0,len(y[i])):
        #     mat[estimated[j]][y[i][j]]+=1
        mat = confusion_matrix(y[i],estimated)
        # auc = roc_auc_score(y[i], estimated, average='samples', sample_weight=None, max_fpr=None)
        acc = balanced_accuracy_score(y[i], estimated, sample_weight=None, adjusted=False)
        conf_mat.append(mat)
        # auc_score.append(auc)
        accuracy.append(acc)
    conf_mat = np.mean(conf_mat,0)
    # auc_score = np.mean(auc_score,0)
    accuracy = np.mean(accuracy,0)
    print('accuracy = ')
    print(accuracy)
    # print('auc = '+ auc_score)
    df_cm = pd.DataFrame(conf_mat, index = [i for i in ['opel', 'saab', 'bus', 'van']],
                  columns = [i for i in ['opel', 'saab', 'bus', 'van']])
    # plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    # return score


