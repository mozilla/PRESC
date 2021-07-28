from presc.copies.sampling import labeling
from presc.copies.evaluations import empirical_fidelity_error


class ClassifierCopy:
    def __init__(
        self,
        original,
        copy,
        sampling_function,
        *sampling_parameters,
        **k_sampling_parameters
    ):
        self.original = original
        self.copy = copy
        self.sampling_function = sampling_function
        self.sampling_parameters = sampling_parameters
        self.k_sampling_parameters = k_sampling_parameters

    def copy_classifier(self):
        # Generate synthetic data
        X_generated = self.sampling_function(
            *self.sampling_parameters, **self.k_sampling_parameters
        )
        df_generated = labeling(X_generated, self.original)

        # Copy the classifier
        self.copy.fit(df_generated.features, df_generated.labels)

    def get_training_data(self, label_col="y"):
        # Random state needs to be fixed to obtain the same training data
        X_generated = self.sampling_function(
            *self.sampling_parameters, **self.k_sampling_parameters
        )
        df_generated = labeling(X_generated, self.original, label_col=label_col)
        return df_generated

    def generate_synthetic_data(
        self, generated_nsamples=100, random_state=None, label_col="y"
    ):
        k_sampling_parameters_gen = self.k_sampling_parameters
        k_sampling_parameters_gen["random_state"] = random_state
        k_sampling_parameters_gen["nsamples"] = generated_nsamples

        X_generated = self.sampling_function(
            *self.sampling_parameters, **k_sampling_parameters_gen
        )
        df_generated = labeling(X_generated, self.original, label_col=label_col)
        return df_generated

    def compute_fidelity_error(self, test_data):
        y_pred_original = self.original.predict(test_data)
        y_pred_copy = self.copy.predict(test_data)

        return empirical_fidelity_error(y_pred_original, y_pred_copy)
