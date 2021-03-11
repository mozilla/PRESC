class Dataset:
    """
    Convenience API for a dataset used with classification model.

    Wraps a pandas DataFrame and provides shortcuts to access feature and label
    columns. It also allows for other columns, eg. computed columns, to be
    included or added later.

    Attributes
    ----------
    df : Pandas DataFrame
    label_col : str
        The name of the column containing the labels
    feature_cols : Array of str
        An array-like of column names corresponding to model features. If not specified,
        all columns aside from the label column will be assumed to be features.
    """

    def __init__(self, df, label_col, feature_cols=None):
        self._df = df
        self._label_col = label_col
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]
        self._feature_cols = feature_cols

    @property
    def size(self):
        return self._df.shape[0]

    @property
    def feature_names(self):
        """Returns the feature names as a list."""
        return list(self._feature_cols)

    @property
    def features(self):
        """Returns the dataset feature columns."""
        return self._df[self._feature_cols]

    @property
    def labels(self):
        """Returns the dataset label column."""
        return self._df[self._label_col]

    @property
    def other_cols(self):
        """Returns the dataset columns other than features and label."""
        other_cols = [
            c
            for c in self._df.columns
            if c != self._label_col and c not in self._feature_cols
        ]
        return self._df[other_cols]

    @property
    def column_names(self):
        """Returns feature and other column names."""
        other_colnames = list(self.other_cols.columns)
        return self.feature_names + other_colnames

    @property
    def df(self):
        """Returns the underlying DataFrame."""
        return self._df

    def subset(self, subset_rows, by_position=False):
        """
        Returns a Dataset corresponding to a subset of this one.

        Parameters
        ----------
        subset_rows :
            Selector for the rows to include in the subset (that can be passed to `.loc` or `.iloc`).
        by_position : bool
            If `True`, `subset_rows` is interpeted as row numbers (used with `.iloc`).
            Otherwise, `subset_rows` is used with `.loc`.
        """
        indexer = "iloc" if by_position else "loc"
        subset_df = getattr(self._df, indexer)[subset_rows]
        return self.__class__(
            subset_df, label_col=self._label_col, feature_cols=self._feature_cols
        )
