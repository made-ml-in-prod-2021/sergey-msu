import os
import click
import pandas as pd
import logging


@click.command('merge_data')
@click.option('--input-paths', multiple=True)
@click.option('--output-path')
def merge_data(input_paths: list, output_path: str):
    logging.critical('WWWWWWWWWWWWWWWWWWWWWWWWW')
    logging.critical(input_paths)
    logging.critical(output_path)
    X = []
    for input_path in input_paths:
        X = pd.read_csv(os.path.join(input_path, 'data.csv'))
    X = pd.concat(X)

    y = X[['target']]
    X = X.drop('target', axis=1)

    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, 'data.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y.csv'), index=False)


if __name__ == '__main__':
    merge_data()
