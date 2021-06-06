import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    """Fixture for DAG bag."""
    return DagBag(dag_folder='dags/', include_examples=False)


def test_train_model_dag_import(dag_bag):
    assert dag_bag.dags is not None
    assert 'train_model' in dag_bag.dags

    dag = dag_bag.dags['train_model']

    assert dag.tasks is not None
    assert len(dag.tasks) == 9
    assert 'begin-train' in dag.task_dict
    assert 'wait-for-features' in dag.task_dict
    assert 'wait-for-target' in dag.task_dict
    assert 'data-prepare' in dag.task_dict
    assert 'data-split' in dag.task_dict
    assert 'model-train' in dag.task_dict
    assert 'model-validate' in dag.task_dict
    assert 'data-clean' in dag.task_dict
    assert 'end-train' in dag.task_dict


def test_train_model_dag_structure(dag_bag):
    dag = dag_bag.dags['train_model']

    structure = {
        'begin-train': ['wait-for-features', 'wait-for-target'],
        'wait-for-features': ['data-prepare'],
        'wait-for-target': ['data-prepare'],
        'data-prepare': ['data-split'],
        'data-split': ['model-train'],
        'model-train': ['model-validate'],
        'model-validate': ['data-clean'],
        'data-clean': ['end-train'],
        'end-train': [],
    }
    _test_dag_structure(dag, structure)


def _test_dag_structure(dag, structure):
    for name, task in dag.task_dict.items():
        downstreams = task.downstream_task_ids
        assert downstreams == set(structure[name])
