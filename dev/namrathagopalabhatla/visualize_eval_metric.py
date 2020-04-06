import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_processing import split_data
from classification_models import Classifier


def generate_data(n=15, dist="gaussian", scale=0.5, prob=0.2):
    """Function to generate data of different types of distributions"""
    if n < 3:
        print("Too few points, minimum n required is n > 2.")
        return None

    X = np.arange(15)
    data = []

    for i in range(len(X)):
        curr_data = []
        curr_data.append(X[i])

        if dist == "gaussian":
            y = np.random.normal(scale=scale, size=n).tolist()
        elif dist == "random":
            y = np.random.rand(n).tolist()
        elif dist == "rayleigh":
            y = np.random.rayleigh(scale, size=n).tolist()
        elif dist == "exponential":
            y = np.random.exponential(scale, size=n).tolist()
        elif dist == "logistic":
            y = np.random.logistic(0, scale, size=n).tolist()
        elif dist == "binomial":
            if prob >= 1 or prob <= 0:
                prob = 0.2
            y = (np.random.binomial(1000, prob, size=n) / 1000).tolist()

        curr_data.extend(y)
        data.append(curr_data)

    return data

def visualize_eval_metric(data=None):
    """Function to visualize a band of min, max and mean of the table provided as data"""
    if data is None:
        print("Data not provided.")
        return

    data = np.array(data)

    x = []
    minimum = []
    maximum = []
    mean = []

    for i in range(data.shape[0]):
        x.append(data[i, 0])
        y = data[i, 1:]

        min_y = np.min(y)
        max_y = np.max(y)
        mean_y = np.mean(y)

        maximum.append(max_y)
        minimum.append(min_y)
        mean.append(mean_y)

    plt.figure(figsize=(10, 10))
    plt.plot(x, maximum, "-go", label="Maximum")
    plt.plot(x, mean, "-bo", label="Mean")
    plt.plot(x, minimum, "-ro", label="Minimum")
    plt.fill_between(x, maximum, minimum, alpha=0.2)
    plt.legend()
    plt.show()
