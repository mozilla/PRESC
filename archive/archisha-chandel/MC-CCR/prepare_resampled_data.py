import argparse
import algorithms
import datasets
import logging
import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path


DEFAULT_ROOT_OUTPUT_PATH = Path(__file__).parent / 'resampled_data'
DEFAULT_DATA_PATH = Path(__file__).parent / 'data'


def prepare(dataset, partition, fold, mode='OVA', output_path=DEFAULT_ROOT_OUTPUT_PATH, energy=0.25,
            cleaning_strategy='translate', selection_strategy='proportional', p_norm=1.0, method='sampling'):
    logging.info('Processing fold %dx%d of dataset "%s"...' % (partition, fold, dataset))

    output_path = Path(output_path) / dataset
    output_path.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_test, y_test) = datasets.load(dataset, partition, fold)

    header = pd.read_csv(
        DEFAULT_DATA_PATH / 'folds' / dataset / ('%s.%d.%d.train.csv' % (dataset, partition, fold))
    ).columns

    if mode == 'OVA':
        logging.info('Training distribution before resampling: %s.' % Counter(y_train))

        X_train, y_train = algorithms.MultiClassCCR(
            energy=energy, cleaning_strategy=cleaning_strategy,
            selection_strategy=selection_strategy, p_norm=p_norm, method=method
        ).fit_sample(X_train, y_train)

        logging.info('Training distribution after resampling: %s.' % Counter(y_train))

        csv_path = output_path / ('%s.%d.%d.train.oversampled.csv' % (dataset, partition, fold))

        pd.DataFrame(np.c_[X_train, y_train]).to_csv(csv_path, index=False, header=header)
    elif mode == 'OVO':
        classes = np.unique(np.concatenate([y_train, y_test]))

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                logging.info('Resampling class %s vs. class %s.' % (classes[i], classes[j]))

                indices = ((y_train == classes[i]) | (y_train == classes[j]))

                X, y = X_train[indices].copy(), y_train[indices].copy()

                logging.info('Training distribution before resampling: %s.' % Counter(y))

                X, y = algorithms.CCR(
                    energy=energy, cleaning_strategy=cleaning_strategy,
                    selection_strategy=selection_strategy, p_norm=p_norm
                ).fit_sample(X, y)

                logging.info('Training distribution after resampling: %s.' % Counter(y))

                csv_path = output_path / ('%s.%d.%d.train.oversampled.%dv%d.csv' %
                                          (dataset, partition, fold, classes[i], classes[j]))

                pd.DataFrame(np.c_[X, y]).to_csv(csv_path, index=False, header=header)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', type=str, choices=datasets.names(), required=True)
    parser.add_argument('-partition', type=int, choices=[1, 2, 3, 4, 5], required=True)
    parser.add_argument('-fold', type=int, choices=[1, 2], required=True)
    parser.add_argument('-mode', type=str, choices=['OVA', 'OVO'], default='OVA')
    parser.add_argument('-output_path', type=str, default=DEFAULT_ROOT_OUTPUT_PATH)
    parser.add_argument('-energy', type=float, default=0.25)
    parser.add_argument('-cleaning_strategy', type=str, choices=['ignore', 'translate', 'remove'], default='translate')
    parser.add_argument('-selection_strategy', type=str, choices=['proportional', 'random'], default='proportional')
    parser.add_argument('-p_norm', type=float, default=1.0)
    parser.add_argument('-method', type=str, choices=['sampling', 'complete'], default='sampling')

    args = parser.parse_args()

    prepare(**vars(args))
