'''This program investigates the effect of the number of folds used in performance 
evaluation on the performance score.  '''
import matplotlib.pyplot as plt
import pandas as pd
import explore_data
from sklearn.neighbors import KNeighborsClassifier
from load_dataset import load_dataset
from vary_folds import vary_folds

filename = "vehicles.csv"
df = load_dataset(filename)

#explore dataset 
print('A number of methods and attributes are used to get a quantitative view of the vehicles.csv dataset.')
explore_data.raw(df)
explore_data.histo(df, 'Class')

target = df['Class']
data = df.drop('Class', axis = 1)

#n = k_nn.tune(target, data)
"""The n_neighbors parameter is kept constant in order to investigate the effect of varying the number of folds.
The arbitrary default n_neighbors=5 is assigned."""
knn = KNeighborsClassifier(n_neighbors=5)

folds, avg_score, duration = vary_folds(knn, data, target)
#Print table
table = pd.DataFrame({'Number of folds':folds, 'Average score':avg_score})
print(table)

#Plot the variation of average score as caused by changing the number of folds
plt.figure()
plt.plot(folds, avg_score, label='Average accuracy')
plt.plot(folds, duration, label='Computation time')
plt.legend()
plt.xlabel('Number of folds')
plt.ylabel('Average accuracy')
plt.show()