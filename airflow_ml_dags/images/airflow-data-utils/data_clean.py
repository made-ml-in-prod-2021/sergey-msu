import os
import logging
import click
import shutil


@click.command('data_clean')
@click.option('--input-paths', multiple=True)
def data_clean(input_paths: list):
    log('Begin clean data folders...')
    for input_path in input_paths:
        if os.path.exists(input_path):
            log(f'Clean resource: {input_path}')
            shutil.rmtree(input_path)
        else:
            log(f'Resourse {input_path} does not exists')
    log('Data folders cleaned successfully')


def log(message):
    logger = logging.getLogger('data_clean')
    logger.warning(message)


if __name__ == '__main__':
    data_clean()
