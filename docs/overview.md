# Overview

PRESC is a toolkit for the evaluation of machine learning classification models.
Its goal is to provide insights into model performance which extend beyond
standard scalar accuracy-based measures and into areas which tend to be
underexplored in applications, including:

- Generalizability of the model to unseen data for which the training set may
  not be representative
- Sensitivity to statistical error and methodological choices
- Performance evaluation localized to meaningful subsets of the feature space
- In-depth analysis of misclassifications and their distribution in the feature
  space

As a tool, PRESC is intended for use by ML engineers to assist in the
development and updating of models.
Given a dataset and machine learning classifer, it runs evaluations covering
different aspects of model performance.
These can be explored individually, eg. in a Jupyter notebook, or they can be
viewed collectively in a standalone graphical report.

An eventual goal is to provide integration into a Continuous Integration
workflow: evaluations would be run as a part of CI, for example, on regular
model updates, and fail if metrics produce unacceptable values.



