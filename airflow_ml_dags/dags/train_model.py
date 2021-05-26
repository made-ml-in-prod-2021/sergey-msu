import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor


from utils import load_args

DAG_NAME = 'train_model'

default_args, dag_args, tasks_args = load_args(__name__, DAG_NAME)
local_data_volume = tasks_args['local_data_volume']

with DAG(DAG_NAME, default_args=default_args, **dag_args) as dag:
    start_task = DummyOperator(task_id='begin_train')

    data_sensor_args = tasks_args['data_sensor']

    wait_for_features = FileSensor(
        task_id='wait_for_features',
        poke_interval=10,
        retries=100,
        filepath=os.path.join(data_sensor_args['input_path'],
                              data_sensor_args['features_file'])
    )
    wait_for_target = FileSensor(
        task_id='wait_for_target',
        poke_interval=10,
        retries=100,
        filepath=os.path.join(data_sensor_args['input_path'],
                              data_sensor_args['target_file'])
    )





    end_task = DummyOperator(task_id='end_train')

    start_task >> [wait_for_features, wait_for_target] >> end_task

    # merge_data_args = tasks_args['merge_data']
    # input_paths = [f'--input-paths \"{p}\" '
    #                for p in merge_data_args["input_paths"]]
    # output_path = f'--output-path \"{merge_data_args["output_path"]}\"'
    # merge_data_command = ''.join(input_paths) + output_path
    # merge_data = DockerOperator(
    #     task_id='merge-data',
    #     image='sergey.polyanskikh/airflow-merge-data',
    #     command=merge_data_command,
    #     **tasks_args['default_args'],
    # )
    #
    # clean_data_args = tasks_args['merge_data']
    # input_paths = [f'--input-paths \"{p}\" '
    #                for p in clean_data_args["input_paths"]]
    # clean_data_command = ''.join(input_paths)
    # clean_data = DockerOperator(
    #     task_id='clean-data',
    #     image='sergey.polyanskikh/airflow-clean-data',
    #     command=clean_data_command,
    #     **tasks_args['default_args'],
    # )

