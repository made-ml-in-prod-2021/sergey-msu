import os
import click


@click.command('merge_data')
@click.option('--input-paths', multiple=True)
def clean_data(input_paths: list):
    for input_path in input_paths:
        os.remove(input_path)


if __name__ == '__main__':
    clean_data()
