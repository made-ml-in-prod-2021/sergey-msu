import os
import numpy as np
import click
from sklearn.datasets import load_iris


@click.command('download_data')
@click.option('--name')
@click.option('--output-path', multiple=True)
@click.option('--seed')
def download_data(name: str, output_path: str, seed: int):
    np.random.seed(seed)

    X, y = load_iris(return_X_y=True, as_frame=True)
    X.values += np.random.random(X.size)
    X['target'] = y

    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, 'data.csv'), index=False)


if __name__ == '__main__':
    download_data()
