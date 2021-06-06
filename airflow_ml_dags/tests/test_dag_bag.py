import sys
import pytest
from airflow.models import DagBag


sys.path.append('dags')


@pytest.fixture()
def dag_bag():
    """Fixture for DAG bag."""
    return DagBag(dag_folder='dags/', include_examples=False)


def test_dagbag_import(dag_bag):
    assert not dag_bag.import_errors
