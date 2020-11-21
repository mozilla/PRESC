"""Prepare the inputs to the PRESC report."""

import pandas as pd

from presc.dataset import Dataset
from presc.model import ClassificationModel

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Better quality plots
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")

# Load the dataset.

df = pd.read_csv("../../datasets/winequality.csv")
df = df.drop(columns=["quality"])

dataset = Dataset(df, label="recommend")
dataset.split_test_train(0.3)

# Set up the model

model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(class_weight="balanced"))])
cm = ClassificationModel(model, dataset, should_train=True)

# Config options (TODO: read from file)
config = {"misclass_rate": {"num_bins": 20}}
