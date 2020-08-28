PRESC Project Roadmap
=====================

This is an overview of the features planned for integration into PRESC.
It is intended to give a high-level description of how these will work and
sample use cases.
Prioritization and implementation details are maintained in the repo
[issues](https://github.com/mozilla/PRESC/issues).

At the core of PRESC is a collection of analyses that can be run on a given
statistical model and dataset to inform the developer on different aspects of
the model's performance and behaviour.
The two main intended uses are a graphical presentation in a report and a
pass/fail determination obtained by comparing the results against threshold
values determined by the user.
In either case, it would be up to the user to decide on a course of action to
correct deficiencies in the model surfaced by these evaluations.

Planned analyses are described below, grouped by theme.
Some of these will lend themselves to multiple possible visualizations or
summaries, while others will be applicable in a single clear way.
As of right now, the details are fleshed out to varying extents.
The first step in developing many of these will be to build a prototype and test
them out against different models and datasets to get an idea of how they
behave.
Related literature or implementations that we are aware of are referenced below.
Contributions that link additional references to related work are welcome.

TODO: input/output structure, separating computation from visualization.

Analyses: Misclassifications
----------------------------

Many common accuracy metrics involve scalar counts or rates computed from the
confusion matrix.
However, the misclassified points themselves carry much more information about
the model behaviour.
They are indicative of failures in the model, and understanding why they were
misclassified can help improve it.

For example:

- Is the misclassification due to the characteristics of the point itself or to
  the model?
    * It may not be surprising for an outlier to get misclassified by most
      reasonable models.
    * A point in an area of high overlap between the classes may get
      misclassified by some candidate models and not by others, depending on
      where the decision boundary lands.
- How different is the distribution of misclassified points in feature space
  from that for correctly classified points?
    * Is there evidence of systematic bias?

These analyses generally apply to the predictions on a test set by a trained
model, such as the final evaluation on a held-out test set or model selection on
a validation set.

### Conditional confusion matrix

The standard confusion matrix lists counts of overall classification results by
class for a test set, and is used to compute scalar metrics such as accuracy,
precision and recall.
PRESC will additionally compute a confusion matrix restricted to different
subsets of the feature space or test set. 
This way, the confusion matrix and related metrics can be viewed as they vary
across the values of a feature.
This is similar to calibration, which considers accuracy as a function of
predicted score.

__Input:__

- Predicted labels for a test set from a trained model
- Scheme for partitioning the test set
    * eg. binning values of a given feature

__Output:__ Confusion matrix (a _m x m_ tuple) for each partition

__Applications:__

- Performance metrics as a function of partitions:
    * Misclassification counts by class
    * Standard accuracy metrics (eg. accuracy, precision, recall)
    * Proportion of misclassified belonging to a specific class
- Deviation of these per-partition values from the value over the entire test
  set

__Type__: Model performance metric

### Conditional feature distributions

In a sense this reverses the conditioning of the conditional confusion matrix. 
We compute the distribution of a feature over the test set restricted to each
cell of the confusion matrix.
This allows us to compare distributions between misclassified and correctly
classified points.

__Input:__

- Predicted labels for a test set from a trained model
- Column of data from the test set
    * eg. values of a feature
    * could also be predicted scores or a function of the features

__Output:__ Distributional representation (eg. value counts, histogram or
density estimate) for each cell in the confusion matrix

__Applications:__

- Feature exploration conditional on test set predicted outcomes
- Assessment of differences between misclassified and correctly classified
  points in terms of their distribution in feature space
    * Within one class, between multiple classes, or relative to the training
      set
    * Evidence of bias in the misclassifications
    * Are misclassifications concentrated in an area of strong overlap between
      the classes in the training set?
    * Are misclassifications clustered, eg. separated from the majority of
      training points of that class?

__Type__: Feature distributions

### Counterfactual accuracy

How much does the performance of an optimal model which correctly classifies a
misclassified point differ from the current model?
This is measured by searching (the parameter space) for the best performing
model, subject to the constraint that a specific point is correctly classified
([Bhatt et al (2020)][1]).

__Input:__

- Trainable model
    * ie. model specification and training set
- Labeled sample point
    * ie. misclassified test set point

__Output:__ Trained model which correctly classifies the sample point

__Applications:__

- Cost measure for correcting misclassifications
- Measure of whether a misclassification is more likely due to its inherent
  characteristics or to the choice of model.
    * If forcing a correct classification substantially decreases accuracy, then
      it is likely an unusual point relative to the training set (ie. an
      influential point in the statistical sense).
    * If the change in accuracy is minimal, the misclassification may be an
      artifact of the methodology used to select the model.

__Type__: Per-sample metric applied to misclassifications

### Class fit

In addition to looking at distributions across misclassifications, it is useful
to have a distributional goodness-of-fit metric for how much a misclassified
point "belongs" to each class.
We compute entropy-based goodness-of-fit between a misclassified point and each
"true" class as represented by the training set.

__Input:__

- Sample point
    * ie. misclassified test set point
- Dataset
    * ie. training set

__Output:__ Scalar goodness-of-fit measure for each class

__Applications:__

- Measure of surprisal for misclassifications
    * Was the point misclassified because it looks much more like a member of its predicted class than its true class?
    * If it fits well in multiple classes, it may be in an area of high overlap
    * If it doesn't fit well in any class, it may be an outlier.

__Type__: Per-sample metric applied to misclassifications


[1]: ../literature/ML_IRL_20_CFA.pdf 
