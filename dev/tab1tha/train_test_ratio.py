import matplotlib.pyplot as plt
from load_dataset import load_dataset
from test_size_vary import vary_size

"""The vehicles.csv dataset is used so that variations can be seen more clearly as 
compared to the totally ideal case of generated.csv"""

filename = "vehicles.csv"
df = load_dataset(filename)

# df is separated into feature and target variables respectively
X = df.drop("Class", axis=1)
y = df["Class"]

# unpack return value of test_size_vary.vary_size
(test_ratio, accuracy, performance) = vary_size(X, y)

print(performance)
# plot graph to visually demonstrate relationship between splitting ratio and prediction accuracy.
plt.figure()
plt.plot(test_ratio, accuracy)
plt.xlabel("test ratio")
plt.ylabel("accuracy")
plt.show()
