import os
import logging
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command('data_split')
@click.option('--input-path')
@click.option('--output-path')
@click.option('--train-size', type=float)
@click.option('--shuffle', type=bool)
def data_split(input_path: str, output_path: str, train_size: float,
               shuffle: bool):
    log('Begin split data...')

    if not os.path.exists(input_path):
        raise ValueError(f'Data path {input_path or "None"} does not exist')

    data_path = os.path.join(input_path, 'data.csv')
    if not os.path.exists(data_path):
        raise ValueError(f'Data file {data_path} does not exist')

    data = pd.read_csv(data_path)
    data_train, data_valid = train_test_split(data, stratify=data['target'],
                                              train_size=train_size,
                                              shuffle=shuffle, random_state=9)
    data_train.to_csv(os.path.join(output_path, 'data_train.csv'), index=False)
    data_valid.to_csv(os.path.join(output_path, 'data_valid.csv'), index=False)

    log('Data split success')


def log(message):
    logger = logging.getLogger('data_split')
    logger.warning(message)


if __name__ == '__main__':
    data_split()
