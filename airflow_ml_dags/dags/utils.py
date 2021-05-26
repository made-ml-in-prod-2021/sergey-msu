import logging
from datetime import timedelta
import yaml
from pkg_resources import resource_filename


def load_args(res_name: str, dag_name: str):
    filename = resource_filename(res_name, f'../configs/{dag_name}.yaml')
    with open(filename, 'r') as stream:
        config = yaml.safe_load(stream)

    default_args = config['default_args']
    default_args['retry_delay'] = \
        timedelta(minutes=default_args['retry_delay'])
    default_args['execution_timeout'] = \
        timedelta(minutes=default_args['execution_timeout'])
    default_args['dagrun_timeout'] = \
        timedelta(minutes=default_args['dagrun_timeout'])

    dag_args = config['dag_args']
    tasks_args = config['tasks_args']

    return default_args, dag_args, tasks_args
