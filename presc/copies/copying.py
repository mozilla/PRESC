from presc.copies.sampling import labeling
from presc.copies.evaluations import empirical_fidelity_error


class ClassifierCopy:
    def __init__(
        self,
        original,
        copy,
        sampling_function,
        label_col="class",
        **k_sampling_parameters
    ):
        self.original = original
        self.copy = copy
        self.sampling_function = sampling_function
        self.label_col = label_col
        self.k_sampling_parameters = k_sampling_parameters

    def copy_classifier(self, get_training_data=False, **k_mod_sampling_parameters):
        # Generate synthetic data
        df_generated = self.generate_synthetic_data(k_mod_sampling_parameters)
        # Copy the classifier
        self.copy.fit(df_generated.features, df_generated.labels)

        if get_training_data:
            return df_generated

    def generate_synthetic_data(
        self, label_col=None, generated_nsamples=None, random_state=None
    ):
        # Random state needs to be fixed to obtain the same training data
        k_sampling_parameters_gen = self.k_sampling_parameters

        if generated_nsamples is not None:
            k_sampling_parameters_gen["nsamples"] = generated_nsamples
        if random_state is not None:
            k_sampling_parameters_gen["random_state"] = random_state

        X_generated = self.sampling_function(k_sampling_parameters_gen)

        if label_col is None:
            df_generated = labeling(X_generated, self.original)
        else:
            df_generated = labeling(X_generated, self.original, label_col=label_col)

        return df_generated

    def compute_fidelity_error(self, test_data):
        y_pred_original = self.original.predict(test_data)
        y_pred_copy = self.copy.predict(test_data)

        return empirical_fidelity_error(y_pred_original, y_pred_copy)
