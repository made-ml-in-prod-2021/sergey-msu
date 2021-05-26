import click


@click.command('data_prepare')
@click.option('--input-path')
@click.option('--output-path')
def data_prepare(input_path: str, output_path: str):
    pass


if __name__ == '__main__':
    data_prepare()
