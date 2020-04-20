import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


ORIGINAL_PATH = Path(__file__).parent / 'data' / 'original'
FOLDS_PATH = Path(__file__).parent / 'data' / 'folds'


def partition(name):
    fold_path = FOLDS_PATH / name

    if not fold_path.exists():
        fold_path.mkdir(parents=True, exist_ok=True)

    original_path = ORIGINAL_PATH / ('%s-full.dat' % name)

    metadata_lines = 0

    with open(original_path) as f:
        for line in f:
            if line.startswith('@'):
                metadata_lines += 1

                if line.startswith('@input'):
                    inputs = [l.strip() for l in line[8:].split(',')]
                elif line.startswith('@output'):
                    outputs = [l.strip() for l in line[8:].split(',')]
            else:
                break

    df = pd.read_csv(original_path, skiprows=metadata_lines, header=None)
    df.columns = inputs + outputs
    df = pd.concat([pd.get_dummies(df[inputs]), df[outputs]], axis=1)

    matrix = df.as_matrix()
    X, y = matrix[:, :-1], matrix[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)

    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        skf.get_n_splits(X, y)
        splits = list(skf.split(X, y))

        for j in range(len(splits)):
            train_index, test_index = splits[j]
            scaler = MinMaxScaler().fit(X[train_index])
            train_set = pd.DataFrame(np.c_[scaler.transform(X[train_index]), y[train_index]])
            test_set = pd.DataFrame(np.c_[scaler.transform(X[test_index]), y[test_index]])

            dfs = {'train': train_set, 'test': test_set}

            for partition_type in ['train', 'test']:
                file_name = name + '.' + str(i + 1) + '.' + str(j + 1) + '.' + partition_type + '.csv'
                path = FOLDS_PATH / name / file_name
                dfs[partition_type].to_csv(path, index=False, header=df.columns)


def load(name, partition, fold):
    partitions = []

    for partition_type in ['train', 'test']:
        path = FOLDS_PATH / name / ('%s.%d.%d.%s.csv' % (name, partition, fold, partition_type))
        df = pd.read_csv(path)
        matrix = df.values
        X, y = matrix[:, :-1], matrix[:, -1]

        partitions.append([X, y])

    return partitions


def names():
    return [path.name.replace('-full.dat', '') for path in ORIGINAL_PATH.iterdir()]


if __name__ == '__main__':
    print('Partitioning...')

    for name in names():
        print('Partitioning %s...' % name)
        partition(name)
