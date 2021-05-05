import pytest
import pandas as pd
from logging import Logger
from fastapi.testclient import TestClient

from src.app import app, core
from src.entities import AppParams, MedicalRequest
from src.entities.app_params import read_app_params
from tests.mocks import NopeLogger


@pytest.fixture(scope="session", autouse=True)
def nope_logger() -> Logger:
    return NopeLogger('nope')


@pytest.fixture(scope="session", autouse=True)
def test_data() -> pd.DataFrame:
    return pd.read_csv('data/heart_test.csv').drop('target', axis=1)


@pytest.fixture(scope="session", autouse=True)
def test_params() -> AppParams:
    app_params = read_app_params('../configs/app_config.yaml')
    MedicalRequest.real_features = app_params.features
    return app_params


@pytest.fixture(scope="session", autouse=True)
def init_core(test_params, nope_logger):
    core.init(test_params, nope_logger)


def test_api_root():
    with TestClient(app) as client:
        response = client.get('/')

        assert response.status_code == 200
        assert response.json() == 'this is entry point of our predictor'


def test_api_touch():
    with TestClient(app) as client:
        response = client.get('/touch')

        assert response.status_code == 200
        assert response.json()


def test_api_prediction(test_data):
    with TestClient(app) as client:
        data_js = test_data.to_dict(orient='records')
        response = client.post('/predict/', json={'data': data_js})
        result = [r['result'] for r in response.json()]

        assert response.status_code == 200
        assert result == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]


def test_api_wrong_features(test_data):
    with TestClient(app) as client:
        test_data.columns.values[0] = 'unknown_columns'
        data_js = test_data.to_dict(orient='records')
        response = client.post('/predict/', json={'data': data_js})

        assert response.status_code == 400


def test_api_shuffled_features(test_data):
    with TestClient(app) as client:
        feat1 = test_data.columns.values[0]
        feat2 = test_data.columns.values[1]
        test_data.columns.values[0] = feat2
        test_data.columns.values[1] = feat1
        data_js = test_data.to_dict(orient='records')
        response = client.post('/predict/', json={'data': data_js})

        assert response.status_code == 400
