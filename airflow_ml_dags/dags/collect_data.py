from datetime import timedelta
import yaml
from pkg_resources import resource_filename

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


filename = resource_filename(__name__, '../configs/collect_data.yaml')
with open(filename, 'r') as stream:
    config = yaml.safe_load(stream)

default_args = config['default_args']
default_args['retry_delay'] = timedelta(minutes=default_args['retry_delay'])
default_args['execution_timeout'] = \
    timedelta(minutes=default_args['execution_timeout'])
dag_args = config['dag_args']
tasks_args = config['tasks_args']


with DAG('collect_data', default_args=default_args, **dag_args) as dag:
    merge_data_args = tasks_args['merge_data']
    input_paths = [f'--input-paths \"{p}\" ' for p in merge_data_args["input_paths"]]
    output_path = f'--output-path \"{merge_data_args["output_path"]}\"'
    merge_data_command = ''.join(input_paths) + output_path
    merge_data = DockerOperator(
        image='sergey.polyanskikh/airflow-merge-data',
        command=merge_data_command,
        network_mode='bridge',
        task_id='merge-data',
        do_xcom_push=False,
        volumes=[
            "/home/sergey/Work/courses/made/semester-2/ml-in-prod/sergey-msu/airflow_ml_dags/data:/data"
        ]
    )

    clean_data_command = f'input-paths={merge_data_args["input_paths"]} '
    clean_data = DockerOperator(
        image='sergey.polyanskikh/airflow-clean-data',
        command=clean_data_command,
        network_mode='bridge',
        task_id='clean-data',
        do_xcom_push=False,
        volumes=[
            "/home/sergey/Work/courses/made/semester-2/ml-in-prod/sergey-msu/airflow_ml_dags/data:/data"
        ]
    )

    sources = ['hive', 'clickhouse', 'mongo']
    for source in sources:
        source_args = tasks_args[f'download_data_{source}']
        source_command = f'--output-path \'{source_args["output_path"]}\' ' \
                         f'--seed {source_args["seed"]}'
        download_data_source = DockerOperator(
            image="sergey.polyanskikh/airflow-download-data",
            command=source_command,
            network_mode="bridge",
            task_id=f"download-data-{source}",
            do_xcom_push=False,
            wait_for_downstream=True,
            auto_remove=True,
            volumes=[
                "/home/sergey/Work/courses/made/semester-2/ml-in-prod/sergey-msu/airflow_ml_dags/data:/data"
            ]
        )

        download_data_source >> merge_data

    merge_data >> clean_data
