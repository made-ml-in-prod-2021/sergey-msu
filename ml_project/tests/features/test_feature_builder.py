import pytest
import numpy as np
import pandas as pd

from src.features.feature_builder import FeatureBuilder, FeatureParams
from tests.data_generator import generate_test_dataframe


@pytest.fixture()
def small_data() -> list:
    return generate_test_dataframe(3, 9)


@pytest.fixture()
def small_data_num_missing() -> list:
    return generate_test_dataframe(3, 9, cols_data={'age': [1, 2, np.nan]})


@pytest.fixture()
def small_data_cat_missing() -> list:
    return generate_test_dataframe(4, 9, cols_data={'sex': [1, 1, 0, np.nan]})


@pytest.fixture()
def med_data() -> list:
    return generate_test_dataframe(10, 9)


@pytest.fixture()
def all_features() -> list:
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


@pytest.fixture()
def cat_features() -> list:
    return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']


@pytest.fixture()
def num_features() -> list:
    return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']


@pytest.fixture()
def feature_params(cat_features, num_features) -> FeatureParams:
    return FeatureParams(categorical_feats=cat_features,
                         numerical_feats=num_features)


@pytest.fixture()
def train_feature_builder(all_features, feature_params) -> FeatureBuilder:
    return FeatureBuilder(all_features, feature_params, 'train')


def test_feature_builder_init(all_features: list,
                              feature_params: FeatureParams):
    builder = FeatureBuilder(all_features, feature_params, 'some_mode')
    assert builder.all_features == all_features
    assert builder.params == feature_params
    assert builder.transformer is not None
    assert builder.mode == 'some_mode'


def test_feature_builder_build_categorical(
        train_feature_builder: FeatureBuilder,
        small_data: pd.DataFrame):
    pipeline = \
        train_feature_builder.build_categorical(train_feature_builder.params)
    data = pipeline.fit_transform(small_data)
    data = np.asarray(data.todense())

    assert data.shape == (3, 35)
    assert set(data.flatten()) == {0, 1}
    assert np.allclose(data[:, 0].flatten(), np.array([1, 0, 0]))
    assert np.allclose(data[:, 3].flatten(), np.array([1, 0, 0]))
    assert np.allclose(data[:, 7].flatten(), np.array([1, 1, 0]))


def test_feature_builder_build_numerical(
        train_feature_builder: FeatureBuilder,
        small_data: pd.DataFrame):
    pipeline = \
        train_feature_builder.build_numerical(train_feature_builder.params)
    data = pipeline.fit_transform(small_data)
    data = np.asarray(data)

    assert data.shape == (3, 14)
    assert np.allclose(data[:, 0].flatten(), np.array([48, 73, 74]))
    assert np.allclose(data[:, 3].flatten(), np.array([152, 159, 140]))
    assert np.allclose(data[:, 9].flatten(),
                       np.array([3.59202847, 2.66759589, 2.19135113]))


def test_feature_builder_build_missing_numerical(
        train_feature_builder: FeatureBuilder,
        small_data_num_missing: pd.DataFrame):
    pipeline = \
        train_feature_builder.build_numerical(train_feature_builder.params)
    data = pipeline.fit_transform(small_data_num_missing)
    data = np.asarray(data)

    assert np.allclose(data[:, 0].flatten(), np.array([1, 2, 1.5]))


def test_feature_builder_build_missing_categorical(
        train_feature_builder: FeatureBuilder,
        small_data_cat_missing: pd.DataFrame):
    pipeline = \
        train_feature_builder.build_categorical(train_feature_builder.params)
    data = pipeline.fit_transform(small_data_cat_missing[['sex']])
    data = np.asarray(data.todense())

    assert np.allclose(data[:, 0:2], np.array([[0, 1],
                                               [0, 1],
                                               [1, 0],
                                               [0, 1]]))


def test_feature_builder_build(
        train_feature_builder: FeatureBuilder,
        med_data: pd.DataFrame):
    pipeline = train_feature_builder.transformer
    data = pipeline.fit_transform(med_data)
    data = np.asarray(data)

    assert data.shape == (4, 22)
    assert np.allclose(data[:, 0].flatten(), np.array([1, 0, 1, 1]))
    assert np.allclose(data[:, 3].flatten(), np.array([0, 1, 1, 0]))
    assert np.allclose(
        data[:, -3].flatten(),
        np.array([0.68045656, 1.61366809, 0.95843267, 2.54229133]))
