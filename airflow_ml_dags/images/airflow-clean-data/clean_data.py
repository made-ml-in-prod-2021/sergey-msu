import click
import shutil


@click.command('clean_data')
@click.option('--input-paths', multiple=True)
def clean_data(input_paths: list):
    for input_path in input_paths:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    clean_data()
