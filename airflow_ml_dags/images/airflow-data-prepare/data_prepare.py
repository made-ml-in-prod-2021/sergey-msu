import os
import logging
import pandas as pd
import click


@click.command('data_prepare')
@click.option('--input-path')
@click.option('--output-path')
@click.option('--mode')
def data_prepare(input_path: str, output_path: str, mode: str):
    log(f'Begin prepare data, mode={mode}...')

    if not os.path.exists(input_path):
        raise ValueError(f'Data path {input_path} does not exist')

    data_path = os.path.join(input_path, 'data.csv')
    if not os.path.exists(data_path):
        raise ValueError(f'Data file {data_path} does not exist')

    if mode != 'pred':
        target_path = os.path.join(input_path, 'target.csv')
        if not os.path.exists(target_path):
            raise ValueError(f'Target file {data_path} does not exist')

    os.makedirs(output_path, exist_ok=True)

    data = pd.read_csv(data_path)
    X = data.copy()
    X.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

    if mode != 'pred':
        y = pd.read_csv(target_path)
        X['target'] = y.values

    X.to_csv(os.path.join(output_path, 'data.csv'), index=False)

    log(f'Data prepared successfully')


def log(message):
    logger = logging.getLogger('data_prepare')
    logger.warning(message)


if __name__ == '__main__':
    data_prepare()
