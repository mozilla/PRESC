class PrescError(ValueError, AttributeError):
    """General exception class for errors related to PRESC computations."""


def include_exclude_list(all_vals, included="*", excluded=None):
    """Find values remaining after inclusions and exclusions are applied.

    Values are first restricted to explicit inclusions, and then exclusions are
    applied.

    all_vals: the full list of possible values
    included: the list of values to include. Those not listed here are dropped.
    excluded: the list of values to drop (after restricting to included).

    The special values "*" and None are interpreted as "all" and "none"
    respectively for `included` and `excluded`.
    """
    if not included or excluded == "*":
        return []

    if included == "*":
        incl_vals = all_vals
    else:
        incl_vals = [x for x in included if x in all_vals]

    if excluded:
        incl_vals = [x for x in incl_vals if x not in excluded]

    return incl_vals
