import os
import pickle
import json
import logging
import pandas as pd
import click
from sklearn.metrics import accuracy_score


@click.command('model_validate')
@click.option('--input-path')
@click.option('--output-path')
def model_validate(input_path: str, output_path: str):
    log('Begin validate model...')

    if not os.path.exists(input_path):
        raise ValueError(f'Data path {input_path} does not exist')
    if not os.path.exists(output_path):
        raise ValueError(f'Model path {output_path} does not exist')

    data_valid_path = os.path.join(input_path, 'data_valid.csv')
    if not os.path.exists(data_valid_path):
        raise ValueError(f'Valid data file {data_valid_path} does not exist')

    model_path = os.path.join(output_path, 'model.pkl')
    if not os.path.exists(model_path):
        raise ValueError(f'Model file {model_path} does not exist')

    log('Load trained model...')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_valid = pd.read_csv(data_valid_path)
    X_valid = data_valid.drop('target', axis=1)
    y_valid = data_valid['target'].values

    log('Validate model...')
    y_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    log(f'Model accuracy:' + str(round(accuracy, 2)))

    with open(os.path.join(output_path, 'valid_stats.json'), 'w') as f:
        json.dump({'accuracy': accuracy}, f)

    log('Model validated successfully')


def log(message):
    logger = logging.getLogger('model_validate')
    logger.warning(message)


if __name__ == '__main__':
    model_validate()
