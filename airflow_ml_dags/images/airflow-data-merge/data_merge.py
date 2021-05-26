import os
import click
import logging
import pandas as pd


@click.command('data_merge')
@click.option('--input-paths', multiple=True)
@click.option('--output-path')
def data_merge(input_paths: list, output_path: str):
    log(f'Begin merge data from different data sources...')

    X = []
    for input_path in input_paths:
        log(f'Load data: {input_path}...')
        x = pd.read_csv(os.path.join(input_path, 'data.csv'))
        X.append(x)
    X = pd.concat(X)

    y = X[['target']]
    X = X.drop('target', axis=1)

    log(f'Save result data to {output_path}...')
    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, 'data.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y.csv'), index=False)

    log(f'Data merged and saved successfully')


def log(message):
    logger = logging.getLogger('data_merge')
    logger.warning(message)


if __name__ == '__main__':
    data_merge()
