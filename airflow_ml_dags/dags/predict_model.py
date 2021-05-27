import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

from utils import load_args


DAG_NAME = 'predict_model'
default_args, dag_args, tasks_args = load_args(__name__, DAG_NAME)

# import airflow
# airflow.utils.email.send_email('spmathf@gmail.com', 'Airflow TEST HERE', 'This is airflow status success')

with DAG(DAG_NAME, default_args=default_args, **dag_args) as dag:
    data_sensor_args = tasks_args['data_sensor']

    wait_for_data = FileSensor(
        task_id='wait_for_data',
        poke_interval=10,
        retries=100,
        filepath=os.path.join(data_sensor_args['input_path'],
                              data_sensor_args['data_file'])
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

    model_predict_args = tasks_args['model_predict']
    model_predict_command = \
        f'--input-path \"{model_predict_args["input_path"]}\" ' \
        f'--output-path \"{model_predict_args["output_path"]}\" ' \
        f'--model-path \"{model_predict_args["model_path"]}\" '
    model_predict = DockerOperator(
        task_id='model-predict',
        image='sergey.polyanskikh/airflow-model-predict',
        command=model_predict_command,
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

    wait_for_data >> data_prepare >> model_predict >> data_clean
