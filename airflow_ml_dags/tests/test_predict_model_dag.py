import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    """Fixture for DAG bag."""
    return DagBag(dag_folder='dags/', include_examples=False)


def test_predict_model_dag_import(dag_bag):
    assert dag_bag.dags is not None
    assert 'predict_model' in dag_bag.dags

    dag = dag_bag.dags['predict_model']

    assert dag.tasks is not None
    assert len(dag.tasks) == 6
    assert 'begin-make-predictions' in dag.task_dict
    assert 'wait-for-model' in dag.task_dict
    assert 'wait-for-data' in dag.task_dict
    assert 'data-prepare' in dag.task_dict
    assert 'model-predict' in dag.task_dict
    assert 'data-clean' in dag.task_dict


def test_predict_model_dag_structure(dag_bag):
    dag = dag_bag.dags['predict_model']

    structure = {
        'begin-make-predictions': ['wait-for-data', 'wait-for-model'],
        'wait-for-data': ['data-prepare'],
        'wait-for-model': ['data-prepare'],
        'data-prepare': ['model-predict'],
        'model-predict': ['data-clean'],
        'data-clean': [],
    }
    _test_dag_structure(dag, structure)


def _test_dag_structure(dag, structure):
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])
