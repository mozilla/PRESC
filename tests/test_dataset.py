from pandas.testing import assert_frame_equal, assert_series_equal

from presc.dataset import Dataset


def test_dataset(dataset_df):
    ds = Dataset(dataset_df, "label")

    assert ds.size == 100
    assert ds.feature_names == ["a", "b", "c", "d", "e"]
    assert ds.column_names == ["a", "b", "c", "d", "e"]
    assert_frame_equal(ds.features, dataset_df.drop(columns=["label"]))
    assert_series_equal(ds.labels, dataset_df["label"])
    assert ds.other_cols.size == 0
    assert ds.df is dataset_df


def test_dataset_other_cols(dataset_df):
    ds = Dataset(dataset_df, "label", feature_cols=["a", "b", "d"])

    assert ds.size == 100
    assert ds.feature_names == ["a", "b", "d"]
    assert ds.column_names == ["a", "b", "d", "c", "e"]
    assert_frame_equal(ds.features, dataset_df[["a", "b", "d"]])
    assert_series_equal(ds.labels, dataset_df["label"])
    assert_frame_equal(ds.other_cols, dataset_df[["c", "e"]])
    assert ds.df is dataset_df

    ds.df["avg_col"] = (dataset_df["a"] + dataset_df["b"]) / 2
    assert_frame_equal(ds.other_cols, dataset_df[["c", "e", "avg_col"]])
    assert ds.feature_names == ["a", "b", "d"]
    assert ds.column_names == ["a", "b", "d", "c", "e", "avg_col"]


def test_subset(dataset_df):
    # Change the index so that labels don't correspond to positions.
    df = dataset_df.set_index(dataset_df.index + 51)
    df["avg_col"] = (df["a"] + df["b"]) / 2
    ds = Dataset(df, "label", feature_cols=["a", "b", "c", "d", "e"])

    ds_C = ds.subset(ds.df["c"] == "C")
    assert isinstance(ds_C, Dataset)
    assert ds_C.size == 15
    assert ds_C.feature_names == ["a", "b", "c", "d", "e"]
    assert ds_C.column_names == ["a", "b", "c", "d", "e", "avg_col"]
    assert_series_equal(ds_C.labels, df.loc[df["c"] == "C", "label"])
    assert list(ds_C.other_cols.columns) == ["avg_col"]
    assert_frame_equal(ds_C.df, df[df["c"] == "C"])

    ii = list(range(51, 61))
    ds_ind = ds.subset(ii)
    assert isinstance(ds_ind, Dataset)
    assert ds_ind.size == 10
    assert ds_ind.feature_names == ["a", "b", "c", "d", "e"]
    assert ds_ind.column_names == ["a", "b", "c", "d", "e", "avg_col"]
    assert_series_equal(ds_ind.labels, df.iloc[range(10)]["label"])
    assert list(ds_ind.other_cols.columns) == ["avg_col"]
    assert_frame_equal(ds_ind.df, df.iloc[range(10)])

    ds_pos = ds.subset(ii, by_position=True)
    assert isinstance(ds_pos, Dataset)
    assert ds_pos.size == 10
    assert ds_pos.feature_names == ["a", "b", "c", "d", "e"]
    assert ds_pos.column_names == ["a", "b", "c", "d", "e", "avg_col"]
    assert_series_equal(ds_pos.labels, df.iloc[ii]["label"])
    assert list(ds_pos.other_cols.columns) == ["avg_col"]
    assert_frame_equal(ds_pos.df, df.iloc[ii])
