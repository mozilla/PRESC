import pandas as pd

def missclassified_data_category_frquency(y_test, y_pred):
    ''' This function plots frequency of missclassified points against the incorrect categories they were predicted for
    parameters: 
    y_test: reverse factorized test values
    y_pred: reverse factorized predicted values '''
    
    missclassified = []
    for pred,test in zip(y_pred, y_test): # assuming that y_test and y_pred is aleady untokenized, if not call untokenize_test_predict_data function
        if pred != test:
            missclassified.append(pred)
    frequency_df =pd.DataFrame(missclassified)
    frequency_df[0].value_counts().plot(kind='barh')
    
def untokenize_test_predict_data(definition, y_test, y_pred):
    ''' this function can be used to reverse factor test and predicted values before using 'missclassified_data_category_frquency' function
    parameters: 
    y_test:  factorized test values
    y_pred: factorized predicted values
    definitions: categories for reverse factorizing'''
    reversefactor = dict(zip(range(len(definitions)+1),definitions))
    y_test = np.vectorize(reversefactor.get)(y_test)
    y_pred = np.vectorize(reversefactor.get)(y_pred)
