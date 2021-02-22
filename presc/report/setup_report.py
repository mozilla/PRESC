"""Prepare the inputs to the PRESC report."""

import pandas as pd

from presc.dataset import Dataset
from presc.model import ClassificationModel

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit

# Better quality plots
from IPython.display import set_matplotlib_formats

import yaml
import sys

set_matplotlib_formats("svg")

# Load the dataset.

df = pd.read_csv("../../datasets/winequality.csv")
df = df.drop(columns=["quality"])
dataset = Dataset(df, label_col="recommend")

splitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=543)
train_ind, test_ind = next(splitter.split(dataset.features))
train_dataset = dataset.subset(train_ind, by_position=True)
test_dataset = dataset.subset(test_ind, by_position=True)

# Set up the model

model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(class_weight="balanced"))])
cm = ClassificationModel(model, train_dataset, retrain_now=True)

config_filename = "./report_config.yml"
if len(sys.argv) == 2:
    config_filename = sys.argv[1]

print(f"Using config file: {config_filename}")
with open(config_filename) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(f"loaded conditional_metric config: {config['conditional_metric']}")
    print(
        f"loaded conditional_distribution config: {config['conditional_distribution']}"
    )
