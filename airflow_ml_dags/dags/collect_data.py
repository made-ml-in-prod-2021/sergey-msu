from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from utils import load_args


DAG_NAME = 'collect_data'

default_args, dag_args, tasks_args = load_args(__name__, DAG_NAME)

with DAG(DAG_NAME, default_args=default_args, **dag_args) as dag:
    merge_data_args = tasks_args['merge_data']
    input_paths = [f'--input-paths \"{p}\" '
                   for p in merge_data_args["input_paths"]]
    output_path = f'--output-path \"{merge_data_args["output_path"]}\"'
    merge_data_command = ''.join(input_paths) + output_path
    merge_data = DockerOperator(
        task_id='merge-data',
        image='sergey.polyanskikh/airflow-merge-data',
        command=merge_data_command,
        **tasks_args['default_args'],
    )

    clean_data_args = tasks_args['merge_data']
    input_paths = [f'--input-paths \"{p}\" '
                   for p in clean_data_args["input_paths"]]
    clean_data_command = ''.join(input_paths)
    clean_data = DockerOperator(
        task_id='clean-data',
        image='sergey.polyanskikh/airflow-clean-data',
        command=clean_data_command,
        **tasks_args['default_args'],
    )

    sources = ['hive', 'clickhouse', 'mongo']
    for source in sources:
        source_args = tasks_args[f'download_data_{source}']
        source_command = f'--output-path \'{source_args["output_path"]}\' ' \
                         f'--seed {source_args["seed"]}'
        download_data_source = DockerOperator(
            task_id=f"download-data-{source}",
            image="sergey.polyanskikh/airflow-download-data",
            command=source_command,
            **tasks_args['default_args'],
        )

        download_data_source >> merge_data

    merge_data >> clean_data
