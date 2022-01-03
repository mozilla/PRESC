from presc.dataset import Dataset
from presc.copies.sampling import labeling
from presc.copies.evaluations import empirical_fidelity_error


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
            `uniform_sampling`, `normal_sampling`...
        balancing_sampler: bool
            Whether the chosen sampling function does class balancing or not.
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
        balancing_sampler=False,
        label_col="class",
        **k_sampling_parameters
    ):
        self.original = original
        self.copy = copy
        self.sampling_function = sampling_function
        self.balancing_sampler = balancing_sampler
        self.label_col = label_col
        self.k_sampling_parameters = k_sampling_parameters

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
        instantiation and then labels them using the original model. If the same
        data needs to be generated then simply use a specific random seed.

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

        if "nsamples" in k_mod_sampling_parameters.keys():
            k_sampling_parameters_gen["nsamples"] = k_mod_sampling_parameters[
                "nsamples"
            ]
        if "random_state" in k_mod_sampling_parameters.keys():
            k_sampling_parameters_gen["random_state"] = k_mod_sampling_parameters[
                "random_state"
            ]

        X_generated = self.sampling_function(**k_sampling_parameters_gen)

        # If the type of sampling function attempts to balance the synthetic
        # dataset, it returns the features AND the labels. Otherwise, it returns
        # only the features, and the labeling function must be called.
        if self.balancing_sampler:
            df_generated = Dataset(X_generated, label_col=self.label_col)
        else:
            df_generated = labeling(
                X_generated, self.original, label_col=self.label_col
            )
        df_generated = labeling(X_generated, self.original, label_col=self.label_col)

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
