from datetime import timedelta
from ruamel import yaml
from pkg_resources import resource_filename

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


filename = resource_filename(__name__, 'configs/download_data.yaml')
with open(filename, 'r') as stream:
    config = yaml.safe_load(stream)

default_args = config['default_args']
default_args['retry_delay'] = timedelta(minutes=default_args['retry_delay'])
dag_args = config['dag_args']
tasks_args = config['tasks_args']


with DAG('download_data', default_args=default_args, **dag_args) as dag:
    merge_data_args = tasks_args['merge_data']
    merge_data_command = f'--input_paths={merge_data_args["input_paths"]} ' \
                         f'--output_path={merge_data_args["output_path"]}'
    merge_data = DockerOperator(
        image='sergeypolyanskikh/airflow-merge-data',
        command=merge_data_command,
        network_mode='bridge',
        task_id='merge-data',
        do_xcom_push=False,
        # volumes=['/Users/sergey.polyanskikh/PycharmProjects/airflow_examples/data:/data']  # !!!!!!
    )

    clean_data_command = f'--input_paths={merge_data_args["input_paths"]} '
    clean_data = DockerOperator(
        image='sergeypolyanskikh/airflow-clean-data',
        command=clean_data_command,
        network_mode='bridge',
        task_id='clean-data',
        do_xcom_push=False,
        # volumes=['/Users/sergey.polyanskikh/PycharmProjects/airflow_examples/data:/data']  # !!!!!!
    )

    sources = ['hive', 'clickhouse', 'mongo']
    for source in sources:
        source_args = tasks_args[f'download_data_{source}']
        source_command = f'--name={source_args["name"]} ' \
                         f'--output_path={source_args["output_path"]} ' \
                         f'--seed={source_args["seed"]}'
        download_data_source = DockerOperator(
            image='sergeypolyanskikh/airflow-download',
            command=source_command,
            network_mode='bridge',
            task_id=f'download-data-{source}',
            do_xcom_push=False,
            # volumes=['/Users/sergey.polyanskikh/PycharmProjects/airflow_examples/data:/data']  # !!!!!!
        )

        download_data_source >> merge_data

    merge_data >> clean_data
