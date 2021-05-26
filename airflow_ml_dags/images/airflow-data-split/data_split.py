import click


@click.command('data_split')
@click.option('--input-path')
@click.option('--output-path')
@click.option('--train-size')
@click.option('--shuffle')
def data_split(input_path: str, output_path: str, train_size: float,
               shuffle: bool):
    pass


if __name__ == '__main__':
    data_split()
