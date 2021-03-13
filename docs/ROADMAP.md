Roadmap
=======

This is an overview of the evaluations planned for integration into PRESC.
It is intended to give a high-level description of how these will work and
sample use cases.
Prioritization and implementation details are maintained in the repo
[issues](https://github.com/mozilla/PRESC/issues).

At the core of PRESC is a collection of evaluations that can be run on a given
statistical model and dataset pair to inform the developer on different
aspects of the model's performance and behaviour.
The two main intended uses are a graphical presentation in a report and the
detection of potential issues by comparing the results against threshold
values determined by the user.
In either case, results will require some degree of interpretation in the
context of the problem domain, and it will be up to the user to decide on a
course of action to correct deficiencies in the model surfaced by these
evaluations.

Planned evaluations are described below, grouped by theme.
Some of these will lend themselves to multiple possible visualizations or
summaries, while others will be applicable in a single clear way.
The first step in developing many of these will be to build a prototype and test
them out against different models and datasets to get an idea of how they
behave.
Related literature or implementations that we are aware of are referenced below.
Contributions that link additional references to related work are welcome.

For each one, we list expected inputs and output structure, as well as the ways
we expect it to be used.
The description focuses on the underlying computation rather than the ways
results should be presented or visualized.
For some of these, we will want to further summarize the outputs, while others
will be reported as is.


Misclassifications
------------------

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

__Application scope:__ These generally apply to the predictions on a test set by
a trained model, such as the final evaluation on a held-out test set or model
selection on a validation set.

### Conditional metrics

This is implemented in the
[conditional_metric](https://github.com/mozilla/PRESC/blob/master/presc/evaluations/conditional_metric.py)
module.

Standard performance metrics such as accuracy, precision and recall are
computed by summmarizing overall differences between predicted and true labels.
PRESC will additionally compute these differences restricted to subsets of the
feature space or test set.
This way, the confusion matrix and related metrics can be viewed as they vary
across the values of a feature.
This is similar to calibration, which considers accuracy as a function of
predicted score.

__Input:__

- Predicted labels for a test set from a trained model
- Scheme for partitioning the test set
    * eg. binning values of a given feature
- Metric
    * function of predicted and true labels

__Output:__ Metric values for each partition

__Applications:__

- Performance metrics as a function of partitions:
    * Misclassification counts by class
    * Standard accuracy metrics (eg. accuracy, precision, recall)
    * Proportion of misclassified belonging to a specific class
- Deviation of these per-partition values from the value over the entire test
  set

__Type__: Model performance metric

### Conditional feature distributions

This is implemented in the
[conditional_distribution](https://github.com/mozilla/PRESC/blob/master/presc/evaluations/conditional_distribution.py)
module.

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

Datapoints can refer to either the original feature space or an
embedding/transformation.

__Output:__ Scalar goodness-of-fit measure for each class

__Applications:__

- Measure of surprisal for misclassifications
    * Was the point misclassified because it looks much more like a member of
      its predicted class than its true class?
    * If it fits well in multiple classes, it may be in an area of high overlap
    * If it doesn't fit well in any class, it may be an outlier.
- Deviation from a baseline distribution computed using the same approach for
  correctly classified points

__Type__: Per-sample metric applied to misclassifications

### Spatial distributions

This is partially implemented in the
[spatial_distribution](https://github.com/mozilla/PRESC/blob/master/presc/evaluations/spatial_distribution.py)
module.

In some cases, it will be helpful to understand where a misclassified point lies
in the feature space in relation to other training points.
While this does not translate to intuition about model behaviour for all types
of model, it can still be useful as a view into the geometry of the feature
space.
PRESC does this by computing the distribution of pairwise distances between a
misclassified point and other training points split by class.

__Input:__

- Sample point
    * ie. misclassified test set point
- Dataset
    * ie. training set
- Metric to measure distances in the feature space
    * eg. Euclidian

Datapoints and metric can refer to either the original feature space or an
embedding/transformation.

__Output:__ Distributional representation (histogram or density estimate) for
each class

__Applications:__

- Geometric class-fit measure for misclassifications
    * Can help to distinguish between misclassifications that are outliers (far
      from all training points), those which lie in an area of high overlap, and
      those which are closer to a different class
- Deviation from a baseline distribution computed using the same approach for
  correctly classified points

__Type__: Per-sample metric applied to misclassifications


Robustness to unseen data
-------------------------

Ideally, our model should perform well for unseen data points for which
predictions are requested.
Standard performance measures, computed by averaging results over a random split
of the data, represent average-case performance for data which is identically
distributed to the training data, an assumption which is often not met in
practice.
Here we consider performance evaluations that account for distributional
differences in training and test sets.

__Application scope:__ These can be applied either as strategies for model
selection or as an evaluation methodology on a final test set.

### Feature-based splitting

While we don't know along which dimensions the unseen data will differ from our
training set, we can take into account possible training set bias by validating
over explicitly biased splits.
These are generated by holding out training points whose values for a given
feature fall in a particular range.

__Input:__

- Dataset
    * ie. training set
- Scheme for partitioning the test set
    * eg. binning values of a given feature

Datapoints and partitioning can refer to either the original feature space or an
embedding/transformation.

__Output:__ Sequence of splits of the dataset holding out one partition each time

__Applications:__

- Model selection using cross-validation taking training set bias into account
- Feature selection using susceptibility to bias as a criterion
- Model performance range estimate in the face of biased data
    * ie. evaluate on test-set data belonging to the held out partition, having
      trained on training data in the other partitions
- Deviation from overall performance metric over the entire test set

__Type__: Dataset splitting scheme for validation

### Entropy-based splitting

Another approach to non-random splitting is to partition in terms of
distributional differences rather than the values of a specific feature.
Here we generate splits achieving a target distributional dissimilarity value
between the train and test portions.

__Input:__

- Dataset
    * ie. training set

Datapoints can refer to either the original feature space or an
embedding/transformation.

__Output:__ Sequence of randomized splits of the dataset achieving target dissimilarity (K-L divergence) values

__Applications:__

- Model selection using cross-validation taking into account robustness to
  unseen data
- Model performance range estimate in the face of data shift
    * ie. select one subset from the training set and another from the test set
- Deviation from overall performance metric over the entire test set

__Type__: Dataset splitting scheme for validation

### Label flipping

How many training labels would have to change for the decision boundary to move?
A more robust model which is not overfit should be able to sustain more label
changes without a significant impact on its performance.
Label flipping could occur in practice, for example, if some of the training
data is mislabeled, or certain areas of the feature space can reasonably belong
to multiple classes.
To measure this, we compute the change in model performance as more and more
training point labels are flipped.

__Input:__

- Training set
- Model specification
- Performance measurement scheme
    * ie. test set and performance metric

__Output:__ Function (sequence of tuples) mapping number of flipped labels to
performance metric values

__Applications:__

- Measure of robustness to mislabeled data
- Measure of overfitting
- Influence measure for points or clusters in the training data

__Type__: Model performance metric

### Novelty

Scenarios in which classification models are deployed generally evolve over
time, and eventually the data used to train the model may no longer be
representative of the cases for which predictions are requested.
PRESC will include functionality to determine how much a new set of labeled
data (if available) has diverged from the current training set.
This will help to inform when a model update is needed.

__Input:__

- Previous dataset
    * ie. current training set
- New dataset
    * ie. newly available labeled data

__Output:__ Similarity measure between the two datasets (scalar or
distributional)

__Applications:__

- Measure of novelty for new labeled data (eg. available as a result of
  ongoing human review)
- Measure of appropriateness of the model on new data
    * eg. improvement in performance on the new data between a model trained
      including the new data and the original model, as a function of novelty
- Decision rule for when to trigger a model update
- Deviation from baseline computed from subsets of the existing training set

__Type__: Dataset comparison metric


Stability of methodology
------------------------

While models are generally selected to maximize some measure of performance,
the final choice of model also carries an inherent dependence on the
methodology used to train it.
For example, if a different train-test split were used, the final model would
likely be slightly different.
These analyses measure the effects of methodological choices on model
performance, with the goal of minimizing them.
Note that, for any of these approaches which uses resampling to estimate
variability, computation cost needs to be taken into account as a part of the
design.

__Application scope:__ These generally require a model specification and
training set, and can be applied post-hoc (to assess error in reported
results) or prior to training (to help select methodology, using an assumed
prior model choice).

### Train-test splitting

This is implemented in the
[train_test_splits](https://github.com/mozilla/PRESC/blob/master/presc/evaluations/train_test_splits.py)
module.

When training a model, a test set is typically held out at the beginning so as
to provide an unbiased estimate of model performance on unseen data.
However, the size of this test set itself influences the quality of this
estimate.
To assess this, we consider the variability and bias in performance metrics as
the test set size varies.
This is estimated by splitting the input dataset at different proportions and
training and testing a model across these.

__Input:__

- Dataset
    * ie. training set
- Model specification
- Performance metric

__Output:__ Function (sequence of tuples) mapping train-test split size to
performance estimates represented as a mean with confidence bounds

__Applications:__

- Measure of error (variance/bias) in test set performance evaluations
- Selection of train-test split size to minimize bias and variance
- Deviation from average-case performance

__Type__: Model performance confidence metric

### Cross-validation folds

Similarly, the choice of validation methodology will influence the quality of
estimates obtained using it.
PRESC assesses this by computing model cross-validation (CV) model performance
estimates across different numbers of folds.

__Input:__

- Dataset
    * ie. training set
- Model specification
- Performance metric

__Output:__ Function (sequence of tuples) mapping number of CV folds to
performance estimates represented as a mean with confidence bounds

__Applications:__

- Measure of error (variance/bias) in CV performance evaluations
- Selection of number of CV folds to minimize bias and variance
- Deviation from average-case performance

__Type__: Model performance confidence metric



[1]: ../literature/ML_IRL_20_CFA.pdf 
