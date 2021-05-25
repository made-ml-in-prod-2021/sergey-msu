import os
import click
import pandas as pd


@click.command('merge_data')
@click.option('--input-paths', multiple=True)
@click.option('--output-path')
def merge_data(input_paths: list, output_path: str):
    X = []
    for input_path in input_paths:
        x = pd.read_csv(os.path.join(input_path, 'data.csv'))
        X.append(x)
    X = pd.concat(X)

    y = X[['target']]
    X = X.drop('target', axis=1)

    os.makedirs(output_path, exist_ok=True)
    X.to_csv(os.path.join(output_path, 'data.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y.csv'), index=False)


if __name__ == '__main__':
    merge_data()
