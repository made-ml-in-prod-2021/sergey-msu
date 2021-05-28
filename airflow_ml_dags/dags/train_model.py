import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor

from utils import load_args


DAG_NAME = 'train_model'

default_args, dag_args, tasks_args = load_args(__name__, DAG_NAME)

with DAG(DAG_NAME, default_args=default_args, **dag_args) as dag:
    start_task = DummyOperator(task_id='begin-train')

    data_sensor_args = tasks_args['data_sensor']

    wait_for_features = FileSensor(
        task_id='wait-for-features',
        poke_interval=10,
        retries=100,
        filepath=os.path.join(data_sensor_args['input_path'],
                              data_sensor_args['features_file'])
    )
    wait_for_target = FileSensor(
        task_id='wait-for-target',
        poke_interval=10,
        retries=100,
        filepath=os.path.join(data_sensor_args['input_path'],
                              data_sensor_args['target_file'])
    )

    data_prepare_args = tasks_args['data_prepare']
    data_prepare_command = \
        f'--input-path \"{data_prepare_args["input_path"]}\" ' \
        f'--output-path \"{data_prepare_args["output_path"]}\" ' \
        f'--mode \"{data_prepare_args["mode"]}\"'
    data_prepare = DockerOperator(
        task_id='data-prepare',
        image='sergey.polyanskikh/airflow-data-prepare',
        command=data_prepare_command,
        **tasks_args['default_args'],
    )

    data_split_args = tasks_args['data_split']
    data_split_command = \
        f'--input-path \"{data_split_args["input_path"]}\" ' \
        f'--output-path \"{data_split_args["output_path"]}\" ' \
        f'--train-size \"{data_split_args["train_size"]}\" ' \
        f'--shuffle \"{data_split_args["shuffle"]}\"'
    data_split = DockerOperator(
        task_id='data-split',
        image='sergey.polyanskikh/airflow-data-split',
        command=data_split_command,
        **tasks_args['default_args'],
    )

    model_train_args = tasks_args['model_train']
    model_train_command = \
        f'--input-path \"{model_train_args["input_path"]}\" ' \
        f'--output-path \"{model_train_args["output_path"]}\"'
    model_train = DockerOperator(
        task_id='model-train',
        image='sergey.polyanskikh/airflow-model-train',
        command=model_train_command,
        **tasks_args['default_args'],
    )

    model_validate_args = tasks_args['model_validate']
    model_validate_command = \
        f'--input-path \"{model_validate_args["input_path"]}\" ' \
        f'--output-path \"{model_validate_args["output_path"]}\" '
    model_validate = DockerOperator(
        task_id='model-validate',
        image='sergey.polyanskikh/airflow-model-validate',
        command=model_validate_command,
        **tasks_args['default_args'],
    )

    data_clean_args = tasks_args['data_clean']
    data_clean_command = f'--input-paths \"{data_clean_args["input_path"]}\"'
    data_clean = DockerOperator(
        task_id='data-clean',
        image='sergey.polyanskikh/airflow-data-clean',
        command=data_clean_command,
        **tasks_args['default_args'],
    )

    end_task = DummyOperator(task_id='end-train')

    start_task >> \
        [wait_for_features, wait_for_target] >> data_prepare >> data_split >> \
        model_train >> model_validate >> data_clean >> \
        end_task
