from presc.alberginia.data_exploration import load_file


def test_load_file():
    fn = "datasets/generated.csv"
    load_file(fn)
