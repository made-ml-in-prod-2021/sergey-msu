import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    """Fixture for DAG bag."""
    return DagBag(dag_folder='dags/', include_examples=False)


def test_collect_data_dag_import(dag_bag):
    assert dag_bag.dags is not None
    assert 'collect_data' in dag_bag.dags

    dag = dag_bag.dags['collect_data']

    assert dag.tasks is not None
    assert len(dag.tasks) == 6
    assert 'begin-collect-data' in dag.task_dict
    assert 'merge-data' in dag.task_dict
    assert 'clean-data' in dag.task_dict
    assert 'data-download-hive' in dag.task_dict
    assert 'data-download-mongo' in dag.task_dict
    assert 'data-download-clickhouse' in dag.task_dict


def test_collect_data_dag_structure(dag_bag):
    dag = dag_bag.dags['collect_data']

    structure = {
        'begin-collect-data': ['data-download-hive',
                               'data-download-mongo',
                               'data-download-clickhouse'],
        'data-download-hive': ['merge-data'],
        'data-download-mongo': ['merge-data'],
        'data-download-clickhouse': ['merge-data'],
        'merge-data': ['clean-data'],
        'clean-data': []
    }
    _test_dag_structure(dag, structure)


def _test_dag_structure(dag, structure):
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])
