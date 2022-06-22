# Machine Learning Classifier Copies


The use of propietary black-box machine learning models and APIs makes it very
difficult to control and mitigate their potential harmful effects (such as lack 
of transparency, privacy safeguards, robustness, reusability or fairness). The 
technique of [Machine Learning Classifier Copying](https://ieeexplore.ieee.org/document/9181566) allows us to build a new model 
that replicates the decision behaviour of an existing one without the need of 
knowing its architecture nor having access to the original training data.

Among the multiple applications of Machine Learning Classifier Copying, a 
systematic construction and examination of model copies has the potential to be 
a universally accessible and inexpensive approach to study and evaluate a rich 
variety of original models, and to help understand their behavior.

An implementation of Machine Learning Classifier Copying has been added to the 
PRESC package, so that this tool becomes readily accessible to researchers and 
practitioners. The solution provides a model agnostic sampling strategy and an 
automated copy process for a number of fundamentally different hypothesis 
spaces, so that the set of achievable copy-model-fidelity measures can be
used as a diagnostic measure of the original model characteristics.



## Copying pipeline

![Scheme](_images/ML-classifier-copying-package-diagram_v2.svg)

To carry out the copy, the `presc.copies.copying.ClassifierCopy` class needs two inputs: the original classifier to copy and, depending on the sampler, a dictionary with basic descriptors of the features. Right now the package assumes that we have the classifier saved as a sklearn-type model. The original data is not necessary to perform the copy but, if available, the `presc.copies.sampling.dynamical_range` function can conveniently extract the basic descriptors of its features into a dictionary. In this case, the data should be available as a pandas DataFrame. Otherwise, the dictionary with the basic feature descriptors can always be built manually. Even if we don't have access to the original data or detailed information of the features, we need at least to be able to make a guess or some assumptions about them.


When instantiating the `presc.copies.copying.ClassifierCopy` class, an instance of a sklearn-type model to build the copy must also be specified, as well as the choice of sampling function and its options. The necessary feature descriptors for the sampler can be the maximum and minimum values that the features can take, like in the case of `presc.copies.sampling.grid_sampling` and `presc.copies.sampling.uniform_sampling`, the mean and standard deviation of each feature, such as in the `presc.copies.sampling.normal_sampling`, or an overall minimum and maximum single value common to all features, as in the case of `presc.copies.sampling.spherical_balancer_sampling`. 

In the case of samplers for categorical data, such as `presc.copies.sampling.categorical_sampling` and `presc.copies.sampling.mixed_data_sampling`, the data descriptor for each categorical feature must be a dictionary of categories that specifies their frequency.

The `presc.copies.copying.ClassifierCopy.copy_classifier` method will generate synthetic data in the feature space using the sampling function and options specified on instantiation, will label it using the original classifier, and then will use it to train the desired copy model. The generated synthetic training data can be saved in this step if needed but it can also be recovered later using the `presc.copies.copying.ClassifierCopy.generate_synthetic_data` method simply using the same random seed.

## Imbalanced problems

When we talk about imbalance in ML Classifier Copies we are not referring to the balance between classes provided by the original dataset, which is in principle not accessible and thus it does not have any effect in the copy. We are referring to the intrinsic properties of the original classifier.

The copy classifier is normally built using generated data randomly sampled from the whole space, hence, this process will normally tend to generate many more samples for classes that are described by the classifier as occupying a much larger hypervolume. Therefore, it will be the generated data used to train the copy classifier which becomes imbalanced.

To tackle this problem, a mechanism has been introduced to force the balance between classes when generating the synthetic data. Such option can be used with any of the sampling functions by setting the `enforce_balance` as `True`.


## Evaluation of the copy

After the copy has been obtained, an evaluation of the copy can be carried out using the  `presc.copies.copying.ClassifierCopy.compute_fidelity_error` and the `presc.copies.copying.ClassifierCopy.replacement_capability` methods. A more complete overview can also be obtained using `presc.copies.copying.ClassifierCopy.evaluation_summary`.

The evaluation methods need data with which to perform the evaluation, so an unlabeled array-like parameter should be specified when calling them. If original test data is available, it can be used as a test for the copy evaluation. Otherwise, synthetic test data can be generated with the `presc.copies.copying.ClassifierCopy.generate_synthetic_data` method simply using another random seed. However, interpretation of the results will of course have a different meaning than with the original test data.

### Empirical Fidelity Error

When performing a copy, we do not aim for an improvement of the original model performace, but to obtain the exact same behavior. Hence, the copy will be of higher quality when it mimics the orginal model exactly, including its misclassifications. To evaluate this, the best metric to use is the Empirical Fidelity Error.

The Fidelity Error captures the loss of copying, and in its general form is the probability that the copy resembles the model. It evaluates the disagreement between the prediction of the original model and the prediction of the copy model through the percentage error of the copy over a given set of data points, taking the predictions of the original model as ground truth. In the ideal case, the fidelity error is zero and the accuracy of the copy is the same as that of the original classifier. 

Since the synthetic dataset is always separable, theoretically it is always possible to achieve zero empirical error, given we choose a copy model with enough capacity. Hence, copying can be in theory carried out without any loss. However, in practice the generated dataset is invariably finite.

A low Empirical Fidelity Error does not guarantee a good copy. In addition to that, the generated dataset to perform de copy must ensure a good coverage of the input space, and any volume imbalance effect needs to be accounted for as well.

### Replacement Capability

Sometimes we do not really need an exact copy of the original model yielding identical predictions, but we just need to obtain a new model that guarantees the same performance as the original. In these cases we are not concerned with mimicking the predicted classification of specific data points as long as the generalization ability of the copy model is good enough. In this scenario, the best way to evaluate the performance of the copy is to use the Replacement Capability.

The Replacement Capability is the ratio between the accuracy of the copy model with respect to the accuracy of the original model, and it quantifies the ability of a copy to replace the original without any loss in performance. This means that the replacement capability will be one if the copy model matches the accuracy of the original, and that it can have a high value even if the performance of the copy model is not very good, as long as the copy is at least as good as the original. 

The Replacement Capability can also yield in some cases values much larger than one if the copy model generalizes better than the original. This is not normally the case, but it might happen if the original model was poorly chosen and the copy model family is better suited to describe the boundary profile of the original problem.

