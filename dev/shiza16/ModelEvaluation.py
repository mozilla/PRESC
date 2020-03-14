import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier



def define_models():
    
    model = []
    model.append(('LogisticRegression', LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=7600)))
    model.append(('KNN', KNeighborsClassifier()))
    model.append(('DecisionTree', DecisionTreeClassifier()))
    model.append(('GaussianNB', GaussianNB()))
    model.append(('SupportVectorMachine', SVC(kernel="rbf", C=1000, gamma=0.0001)))
    model.append(('RandomForest', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)))
    
    return model

def Evaluation_model(data, X ,Y):
    
    X = data.drop(["Class", "Class_code"], axis=1)
    Y = data["Class"]
    # prepare configuration for cross validation test harness

    models = define_models()
    accuracy_score= []
    models_name = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=10)
        cv_score = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        accuracy_score.append(cv_score)
        models_name.append(name)
        #print(name , " with accuracy: ", cv_score.mean(), " and standard deviation  " , cv_score.std())
        
    box_plot(accuracy_score , models_name)

def box_plot(accuracy_score , models_name):
    fig = plt.figure(figsize = (18,15))
    ax = fig.add_subplot(111)
    boxplots = ax.boxplot(accuracy_score,
           notch = True,
           labels= models_name ,
           widths = .7,
           patch_artist=True,
           medianprops = dict(linestyle='-', linewidth=2, color='Yellow'),
           boxprops = dict(linestyle='--', linewidth=2, color='Black', facecolor = 'green', alpha = .4)
          );

    boxplot1 = boxplots['boxes'][4]
    boxplot1.set_facecolor('red')

    plt.xlabel('Models', fontsize = 20);
    plt.ylabel('Accuracy', fontsize = 20);
    plt.xticks(fontsize = 16);
    plt.yticks(fontsize = 16);
    






def Logistic_Regression(X, y):
    """ Tuning Logistic Regression to increase the accuracy of model """

    parameters = {
        "penalty": ["l1", "l2"],
        "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    LR_grid = GridSearchCV(LogisticRegression(), param_grid=parameters, cv=5)
    LR_grid = LR_grid.fit(X, y)
    classifier = LogisticRegression(
              penalty=LR_grid.best_estimator_.get_params()["penalty"],
              C=LR_grid.best_estimator_.get_params()["C"],
              )
    return classifier.fit(X, y)

def SVM_train(X, y):

    """ SVM Classifier"""
    # kernel =' poly ' is taking infinte time that's why it is not added.
    params_grid = [{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]}]

    svm_grid = GridSearchCV(SVC(), params_grid, cv=5)
    svm_grid = svm_grid.fit(X, y)
    classifier = SVC(
        kernel=svm_grid.best_estimator_.kernel,
        C=svm_grid.best_estimator_.C,
        gamma=svm_grid.best_estimator_.gamma,
    )

    return classifier.fit(X, y)

