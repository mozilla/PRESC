"""Example script demonstrating how to run the PRESC report."""

import pandas as pd

from presc.dataset import Dataset
from presc.model import ClassificationModel
from presc.report.runner import ReportRunner

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit

from pathlib import Path

THIS_DIR = Path(__file__).parent
DATASET_DIR = THIS_DIR / ".." / ".." / "datasets" / "winequality.csv"

# Load the dataset.

df = pd.read_csv(DATASET_DIR)
df = df.drop(columns=["quality"])
dataset = Dataset(df, label_col="recommend")

splitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=543)
train_ind, test_ind = next(splitter.split(dataset.features))
train_dataset = dataset.subset(train_ind, by_position=True)
test_dataset = dataset.subset(test_ind, by_position=True)

# Set up the model

model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(class_weight="balanced"))])
cm = ClassificationModel(model)
cm.train(train_dataset)

presc_report = ReportRunner()
presc_report.run(model=cm, test_dataset=test_dataset, train_dataset=train_dataset)

print(f"The report is available at {presc_report.report_html}")
presc_report.open()
