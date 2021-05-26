import click


@click.command('model_train')
@click.option('--input-path')
@click.option('--model-file')
def model_train(input_path: str, model_file: str):
    pass


if __name__ == '__main__':
    model_train()
