from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from utils import load_args


DAG_NAME = 'collect_data'

default_args, dag_args, tasks_args = load_args(__name__, DAG_NAME)

with DAG(DAG_NAME, default_args=default_args, **dag_args) as dag:
    data_merge_args = tasks_args['data_merge']
    input_paths = [f'--input-paths \"{p}\" '
                   for p in data_merge_args["input_paths"]]
    output_path = f'--output-path \"{data_merge_args["output_path"]}\"'
    data_merge_command = ''.join(input_paths) + output_path
    data_merge = DockerOperator(
        task_id='merge-data',
        image='sergey.polyanskikh/airflow-data-merge',
        command=data_merge_command,
        **tasks_args['default_args'],
    )

    data_clean_args = tasks_args['data_clean']
    input_paths = [f'--input-paths \"{p}\" '
                   for p in data_clean_args["input_paths"]]
    data_clean_command = ''.join(input_paths)
    data_clean = DockerOperator(
        task_id='clean-data',
        image='sergey.polyanskikh/airflow-data-clean',
        command=data_clean_command,
        **tasks_args['default_args'],
    )

    sources = ['hive', 'clickhouse', 'mongo']
    for source in sources:
        source_args = tasks_args[f'data_download_{source}']
        source_command = f'--name {source_args["name"]} ' \
                         f'--output-path \'{source_args["output_path"]}\' ' \
                         f'--seed {source_args["seed"]}'
        data_download_source = DockerOperator(
            task_id=f"data-download-{source}",
            image="sergey.polyanskikh/airflow-data-download",
            command=source_command,
            **tasks_args['default_args'],
        )

        data_download_source >> data_merge

    data_merge >> data_clean
