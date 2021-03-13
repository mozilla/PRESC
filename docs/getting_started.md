# Getting started

The core functionality of PRESC is to investigate the performance of a machine
learning model using different evaluation methods.
This is similar to the use of standard performance metrics such as accuracy, but
the PRESC evaluations allow for more granular feedback.
Rather than a single scalar value, the outputs of the evaluations to be
functions or distributions.
Evaluations can be run individually, eg. in a Jupyter notebook, or as a part of
a collective report.

A full example including the code snippets below is available in the
[project repository](https://github.com/mozilla/PRESC/blob/master/examples/report/sample_report.py).


## Inputs

PRESC is designed to be applied to a dataset and pretrained machine learning
classifier.

### Dataset

A user's dataset should be wrapped in a `presc.dataset.Dataset` prior to being
used in evaluations.
This is mainly used to provide uniform access to feature and label columns.

Currently, an input dataset must be a Pandas DataFrame containing named columns
for the model features and the sample labels.
The project repository includes a few well-known
[sample datasets](https://github.com/mozilla/PRESC/tree/master/datasets).
Here is an example using the `winequality.csv` dataset from that directory:

```python
import pandas as pd
from presc.dataset import Dataset

df = pd.read_csv("winequality.csv")
# Drop the `quality` column as it has been binarized as `recommend`
# in this version of the dataset.
df = df.drop(columns=["quality"])

dataset = Dataset(df, label_col="recommend")
```

### Classifier

The evaluations generally operate by querying the classifer for predictions on a
test set, and in some cases by retraining it under different conditions.
The input classifier should be wrapped in a `presc.model.ClassificationModel`.

Currently, PRESC is designed to work with the
[scikit-learn](https://scikit-learn.org/stable/index.html)
machine learning framework.
The input classifier should be a `scikit-learn` Classifier instance, such as
`sklearn.linear_model.LogisticRegression`.
It should be pretrained on a training dataset prior to running the evaluations.
If necessary, this can be done from the `ClassificationModel` instance using a
`Dataset` instance as input.

```python
from sklearn.linear_model import LogisticRegression
from presc.model import ClassificationModel

clf = LogisticRegression()
cm = ClassificationModel(clf)

# Split the wine dataset into train and test portions and train the model.
splitter = ShuffleSplit(test_size=0.3).split(dataset.features)
train_ind, test_ind = next(splitter)
# dataset.subset() returns another Dataset instance
train_dataset = dataset.subset(train_ind, by_position=True)
test_dataset = dataset.subset(test_ind, by_position=True)

cm.train(train_dataset)
```

Note that some preprocessing of the data, such as scaling, is often needed to
achieve meaningful model fits.
It is generally recommended to bundle these together with the classifier using a
`sklearn.pipeline.Pipeline`, so that the same preprocessing will get applied any
time the classifer is retrained.
In this case, the `ClassificationModel` wrapper should be applied to the
`Pipeline` instance.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pl = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
cm = ClassificationModel(pl)
```

## Report

With the inputs prepared, you can now run the PRESC report.
By default, it runs all available evaluations, but its behaviour can be
configured by setting option values or by specifying a configuration file.
The report is built using
[Jupyter Book](https://jupyterbook.org/intro.html)
to execute evaluations from Jupyter notebooks and render them as HTML pages.

The report is managed by a `presc.report.runner.ReportRunner` instance pointing
to the desired output directory. The report will be written to
`<output_dir>/presc_report/`.
By default, it is executed out of a temporary directory, but this can be changed
in order to access the execution artifacts for debugging purposes.
Additionally, default configuration settings can be overridden by passing a path
to a YAML file in the appropriate format.

The report is run by calling the `run()` method on a `ClassificationModel` and
`Dataset`.
Evaluations generally require a test dataset to run on.
Some may use a training dataset as well, although this is not required to run
the report (although, if not specified, evaluations that require this will
fail).
Settings can also be overridden by passing a dict of option values to `run()`.

```python
from presc.report.runner import ReportRunner

report = ReportRunner(output_path="./output", config_filepath="my_config.yml")
# This may take a few minutes
report.run(model=cm, test_dataset=test_dataset, train_dataset=train_dataset)
```

Once completed, the path to the report main page is accessible using the
`report_html` attribute, and the `open()` method will attempt to open it in the
default browser.

```python
# Show the path to the report main page.
print(report.report_html)
# Open in browser.
report.open()
```

The path to the the Jupyter Book build log is accessible from the `jb_build_log`
attribute. This can be useful in diagnosing problems.
