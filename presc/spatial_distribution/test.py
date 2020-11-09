import categorical_distance as catd
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.linear_model import LogisticRegression


def rotate(data, degree):
    # data: M x 2
    theta = np.pi / 180 * degree
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )  # rotation matrix
    return np.dot(data, R.T)


def generate_data(M, var1, var2, degree):
    # M (scalar): The number of data samples
    # var1 (scalar): variance of a
    # var2 (scalar): variance of b
    mu = [0, 0]
    Cov = [[var1, 0], [0, var2]]
    data = np.random.multivariate_normal(mu, Cov, M)
    # shape: M x 2
    # Step II: rotate data by 45 degree counter-clockwise,
    # so that the two dimensions are in fact correlated
    data = rotate(data, degree)
    return data


def main():
    """ jupyter notebook stuff  """
    # Encode y label
    rooms = pd.read_csv(
        "C:/Users/castromi/Documents/GitHub/PRESC/datasets/mushrooms.csv"
    )
    y = rooms.iloc[:, 0].values
    yencoder = LabelEncoder()
    y = yencoder.fit_transform(y)

    # One hot encode X
    X = rooms.drop("edibility", axis=1)
    xencoder = OneHotEncoder()
    X = xencoder.fit_transform(X).toarray()

    # split data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.05, random_state=0
    )

    # run a simple cross validation test
    classifier = LogisticRegression(random_state=0, solver="lbfgs")
    cv = ShuffleSplit(n_splits=5, train_size=0.05)
    scores = cross_validate(classifier, X, y, cv=cv)

    print(" The scores are:\n", scores["test_score"])

    # train our model
    classifier.fit(x_train, y_train)
    # y_pred = classifier.predict(x_test)

    y_pred = classifier.predict(X)

    """ End of jupyternotebook stuff
    M = 5000
    var1 = 1
    var2 = 0.8
    degree = 45




    rooms.head()
    M=len(rooms)
    var1 = 1
    var2 = 0.8
    var3=0.3
    var4=0.9
    degree = 45

    data1 = np.array(generate_data(M, var1, var2, degree))
    data2 = np.array(generate_data(M, var2, var3, degree))
    data1x=[data1[i][0] for i in range(0,len(data1))]
    data1y=[data1[i][1] for i in range(0,len(data1))]
    data2x=[data2[i][0] for i in range(0,len(data2))]
    data2y=[data2[i][1] for i in range(0,len(data2))]



    rooms['data1_x']=data1x
    rooms['data1_y']=data1y
    rooms['data2_x']=data2x
    rooms['data2_y']=data2y

    numrooms = rooms.select_dtypes(include='number')

    print(numrooms.head(900))
    """
    spatdis = catd.SpatialDistribution(rooms, y_pred, y)
    # lin = spatdis.lin(spatdis._data.iloc[-1], spatdis._data.iloc[100])
    # lin2 = spatdis.lin(spatdis._data.iloc[-100], spatdis._data.iloc[100])

    # spatdis.array_of_distance(spatdis._data.iloc[-1],'overlap')
    # spatdis.plot_knearest_points(spatdis._data.iloc[-100], "overlap","goodall3","lin",8000)
    spatdis.plot_full_histogram_report(
        distance_sample=0.0001, mdistance_sample=0.001, histo_sample=50
    )


main()
