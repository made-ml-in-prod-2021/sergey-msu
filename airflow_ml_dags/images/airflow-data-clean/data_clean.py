import os
import click
import shutil


@click.command('data_clean')
@click.option('--input-paths', multiple=True)
def data_clean(input_paths: list):
    for input_path in input_paths:
        if os.path.exists(input_path):
            shutil.rmtree(input_path)


if __name__ == '__main__':
    data_clean()
