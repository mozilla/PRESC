from pandas import cut, qcut
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def get_bins(s, num_bins, quantile=False):
    """Split a Series into discrete bins.

    Parameters
    ----------
    s : pandas Series
    num_bins : int
        The number of bins to split the range of `s` into.
    quantile : bool
        If True, bin edges will correspond to quantiles for
        equally-spaced probabilities. Otherwise, bins are equally spaced on the
        original scale.

    Returns
    -------
    Series
        Series of the same length as `s` indicating the bin for each value,
        as well as an array of bin edges of length `num_bins+1`.
    """
    if quantile:
        # TODO this will fail if a lot of data values are repeated
        # (pseudo-discrete).
        # Can handle this by jittering the values before binning by an
        # appropriate amount (eg. some fraction of the sd).
        return qcut(s, q=num_bins, retbins=True, duplicates="drop")
    else:
        return cut(s, bins=num_bins, retbins=True)


def is_discrete(s):
    """
    Returns
    -------
    bool
        True if the given Series should be considered discrete/categorical."""
    return is_bool_dtype(s) or not is_numeric_dtype(s)
