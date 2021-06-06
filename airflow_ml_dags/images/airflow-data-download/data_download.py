import os
import time
import logging
import numpy as np
import click
from sklearn.datasets import load_iris


@click.command('data_download')
@click.option('--name')
@click.option('--output-path')
@click.option('--seed', type=int)
def data_download(name: str, output_path: str, seed: int):
    log(f'Begin download data from: {name.upper()}...')
    np.random.seed(seed)

    X, y = load_iris(return_X_y=True, as_frame=True)
    time.sleep(seed % 3)
    X.values[:, :] = X.values + np.random.random(X.shape)
    X['target'] = y

    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, 'data.csv'), index=False)

    log(f'Data from: {name.upper()} downloaded successfully')


def log(message):
    logger = logging.getLogger('data_download')
    logger.warning(message)


if __name__ == '__main__':
    data_download()
