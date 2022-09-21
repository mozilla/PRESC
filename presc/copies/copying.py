from presc.dataset import Dataset
from presc.copies.sampling import mixed_data_sampling, labeling, sampling_balancer
from presc.copies.evaluations import (
    empirical_fidelity_error,
    replacement_capability,
    summary_metrics,
)


class ClassifierCopy:
    """Represents a classifier copy and its associated sampling method of choice.

    Each instance wraps the original ML classifier with a ML classifier copy and
    the sampling method to carry out the copy. Methods allow to carry out
    the copy the original classifier, evaluate the quality of the copy, and to
    generate additional data using the original classifier with the sampling
    method specified on instatiation.

    Attributes
    ----------
        original : sklearn-type classifier
            Original ML classifier to be copied.
        copy : sklearn-type classifier
            ML classifier that will be used for the copy.
        sampling_function : function
            Any of the sampling functions defined in PRESC: `grid_sampling`,
            `uniform_sampling`, `normal_sampling`... The `balancing sampler` can
            only be used if the feature space does not contain any categorical
            variable.
        post_sampling_labeling: bool
            Whether generated data must be labeled after the sampling or not.
            If the chosen sampling function already does class labeling (such as
            balancing samplers) then it should be set to False. If the parameter
            enforce_balance is set to True then this parameter does not have any
            effect.
        enforce_balance : bool
            Force class balancing for sampling functions that do not normally
            carry it out intrinsically.
        label_col : str
            Name of the label column.
        **k_sampling_parameters :
            Parameters needed for the `sampling_function`.
    """

    def __init__(
        self,
        original,
        copy,
        sampling_function,
        post_sampling_labeling=True,
        enforce_balance=False,
        label_col="class",
        **k_sampling_parameters
    ):
        self.original = original
        self.copy = copy
        self.sampling_function = sampling_function
        self.post_sampling_labeling = post_sampling_labeling
        if enforce_balance:
            self.post_sampling_labeling = False
        self.enforce_balance = enforce_balance
        self.label_col = label_col
        self.k_sampling_parameters = k_sampling_parameters
        if "random_state" in self.k_sampling_parameters.keys():
            self.random_state = self.k_sampling_parameters["random_state"]
        else:
            self.random_state = None

    def copy_classifier(self, get_training_data=False, **k_mod_sampling_parameters):
        """Copies the classifier using data generated with the original model.

        Generates synthetic data using only basic information of the features
        (dynamic range, mean and sigma), labels it using the original model,
        and trains the copy model with this synthetic data. It can also return
        the generated synthetic data used for training.

        Parameters
        ----------
        get_training_data : bool
            If `True` this method returns the synthetic data generated from the
            original classifier that was used to train the copy.
        **k_mod_sampling_parameters :
            If the "nsamples" and/or "random_state" parameters of the sampling
            function have to be changed in order to obtain a different set of
            synthetic data, they can be specified here.

        Returns
        -------
        presc.dataset.Dataset
            Outputs a PRESC Dataset with the training samples and their labels
            (if `get_training_data` set to `True`).
        """
        # Generate synthetic data
        df_generated = self.generate_synthetic_data(**k_mod_sampling_parameters)
        # Copy the classifier
        self.copy.fit(df_generated.features, df_generated.labels)

        if get_training_data:
            return df_generated

    def generate_synthetic_data(self, **k_mod_sampling_parameters):
        """Generates synthetic data using the original model.

        Generates samples following the sampling strategy specified on
        instantiation for the numerical features and a discrete distribution for
        the categorical features, and then labels them using the original model.
        If the same data needs to be generated then simply use a specific
        random seed.

        Parameters
        ----------
        **k_mod_sampling_parameters :
            If the "nsamples" and/or "random_state" parameters of the sampling
            function have to be changed in order to obtain a different set of
            synthetic data, they can be specified here.

        Returns
        -------
        presc.dataset.Dataset
            Outputs a PRESC Dataset with the generated samples and their labels.
        """
        # Random state needs to be fixed to obtain the same training data
        k_sampling_parameters_gen = self.k_sampling_parameters.copy()

        # Update sampling parameters which have been specified on calling the method
        k_sampling_parameters_gen.update(k_mod_sampling_parameters)

        if self.enforce_balance:
            # Call balancer generating function with sampling parameters
            # (sampling_balancer returns a pandas dataframe)
            X_generated = sampling_balancer(
                original_classifier=self.original, **k_sampling_parameters_gen
            )
        else:
            # Call generating function with sampling parameters
            # (mixed_data_sampling returns a pandas dataframe)
            X_generated = mixed_data_sampling(**k_sampling_parameters_gen)

        # If the type of sampling function attempts to balance the synthetic
        # dataset, it returns the features AND the labels. Otherwise, it returns
        # only the features, and the labeling function must be called.
        if self.post_sampling_labeling:
            df_generated = labeling(
                X_generated, self.original, label_col=self.label_col
            )
        else:
            df_generated = Dataset(X_generated, label_col=self.label_col)

        return df_generated

    def compute_fidelity_error(self, test_data):
        """Computes the empirical fidelity error of the classifier copy.

        Quantifies the resemblance of the copy to the original classifier. This
        value is zero when the copy makes exactly the same predictions than the
        original classifier (including misclassifications).

        Parameters
        ----------
        test_data : array-like
            Dataset with the unlabeled samples to evaluate the resemblance of
            the copy to the original classifier.

        Returns
        -------
        float
            The numerical value of the empirical fidelity error of the copy with
            this dataset.
        """
        y_pred_original = self.original.predict(test_data)
        y_pred_copy = self.copy.predict(test_data)

        return empirical_fidelity_error(y_pred_original, y_pred_copy)

    def replacement_capability(self, test_data):
        """Computes the replacement capability of a classifier copy.

        Quantifies the ability of the copy model to substitute the original
        model, i.e. maintaining the same accuracy in its predictions. This value
        is one when the accuracy of the copy model is the same as the original
        model, although the individual predictions may be different, approaching
        zero if the accuracy of the copy is much smaller than the original, and
        it can even take values larger than one if the copy model is better than
        the original.

        Parameters
        ----------
        test_data : presc.dataset.Dataset
            Subset of the original data reserved to evaluate the resemblance of
            the copy to the original classifier. Or synthetic data generated
            from the original model with the same purpose.

        Returns
        -------
        float
            The numerical value of the replacement capability.
        """
        y_pred_original = self.original.predict(test_data.features)
        y_pred_copy = self.copy.predict(test_data.features)
        return replacement_capability(test_data.labels, y_pred_original, y_pred_copy)

    def evaluation_summary(self, test_data=None, synthetic_data=None):
        """Computes several metrics to evaluate the classifier copy.

        Summary of metrics that evaluate the quality of a classifier copy, not
        only to assess its performance as classifier but to quantify its
        resemblance to the original classifier. Accuracy of the original and the
        copy models (using the original test data), and the empirical fidelity
        error and replacement capability of the copy (using the original test
        data and/or the generated synthetic data). This is a wrapper of the
        `summary_metrics` function applied to the copy and original models in
        this instance.

        Parameters
        ----------
        original_model : sklearn-type classifier
            Original ML classifier to be copied.
        copy_model : presc.copies.copying.ClassifierCopy
            ML classifier copy from the original ML classifier.
        test_data : presc.dataset.Dataset
            Subset of the original data reserved for testing.
        synthetic_data : presc.dataset.Dataset
            Synthetic data generated using the original model.
        show_results : bool
            If `True` the metrics are also printed.

        Returns
        -------
        dict
            The values of all metrics.
        """
        results = summary_metrics(
            original_model=self.original,
            copy_model=self,
            test_data=test_data,
            synthetic_data=synthetic_data,
            show_results=True,
        )
        return results
