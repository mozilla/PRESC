from presc.utils import include_exclude_list


def test_include_exclude():
    vals = ["a", "b", "c", "d", "e"]

    assert include_exclude_list(vals) == vals
    assert include_exclude_list(vals, included=vals) == vals
    assert include_exclude_list(vals, included=None) == []
    assert include_exclude_list(vals, excluded="*") == []
    assert include_exclude_list(vals, excluded=vals) == []

    assert include_exclude_list(vals, included=["c", "b", "e", "y"]) == ["c", "b", "e"]
    assert include_exclude_list(vals, excluded=["a", "e", "z"]) == ["b", "c", "d"]
    assert include_exclude_list(
        vals, included=["c", "b", "e", "y"], excluded=["e"]
    ) == ["c", "b"]
    assert include_exclude_list(vals, included=["c", "b", "e", "y"], excluded="*") == []
    assert include_exclude_list(vals, included=None, excluded=["a"]) == []
