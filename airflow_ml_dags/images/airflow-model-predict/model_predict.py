import os
import pickle
import logging
import pandas as pd
import click


@click.command('model_predict')
@click.option('--input-path')
@click.option('--output-path')
@click.option('--model-path')
def model_predict(input_path: str, output_path: str, model_path: str):
    log('Begin inference...')

    if not os.path.exists(input_path):
        raise ValueError(f'Data path {input_path} does not exist')
    if not os.path.exists(model_path):
        raise ValueError(f'Model path {model_path or "None"} does not exist')
    if not os.path.exists(model_path):
        raise ValueError(f'Model file {model_path} does not exist')

    data_path = os.path.join(input_path, 'data.csv')
    if not os.path.exists(input_path):
        raise ValueError(f'Data file {data_path} does not exist')

    os.makedirs(output_path, exist_ok=True)

    log('Load trained model...')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    data_pred = pd.read_csv(data_path)
    y_pred = model.predict(data_pred)
    y_pred = pd.DataFrame({'pred': y_pred})
    y_pred.to_csv(os.path.join(output_path, 'predictions.csv'))

    log('Inference success')


def log(message):
    logger = logging.getLogger('model_predict')
    logger.info(message)


if __name__ == '__main__':
    model_predict()
