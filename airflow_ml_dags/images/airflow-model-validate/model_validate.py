import click


@click.command('model_validate')
@click.option('--input-path')
@click.option('--output-path')
@click.option('--model-file')
def model_validate(input_path: str, output_path: str, model_file: str):
    pass


if __name__ == '__main__':
    model_validate()
