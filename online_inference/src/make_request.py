""" Simple request/response test script. """
import os
import pandas as pd
import click
import requests

from src.entities.req_params import read_req_params
from src.utils import create_logger


@click.command(name='main')
@click.argument('config-path', default='../configs/req_config.yaml')
def main(config_path: str):
    # read app config
    req_params = read_req_params(config_path)
    logger = create_logger('inference', req_params.logging)

    # load data
    data_path = os.path.abspath(req_params.test_data_path)
    logger.info(f'read data: {data_path}')
    data = pd.read_csv(data_path).drop('target', axis=1)

    # make request
    url = f'http://{req_params.host}:{req_params.port}/predict/'
    logger.info(f'make request to: {url}')
    data_js = data.to_dict(orient='records')
    response = requests.post(url, json={'data': data_js})

    # read response
    logger.info(f'response status: {response.status_code}')
    if response.status_code == 200:
        logger.info(f'response: {response.json()}')

    logger.info(f'done')


if __name__ == "__main__":
    main()
