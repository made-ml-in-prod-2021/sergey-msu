import os
import json
import pytest
import numpy as np
import pandas as pd
from logging import Logger
from tests.mocks import NopeLogger
from src.models.model_trainer import ModelTrainer
from src.features.feature_builder import FeatureBuilder
from src.entities import (TrainingParams,
                          DataParams,
                          SplittingParams,
                          FeatureParams,
                          ConfigTrainParams,
                          AppParams,
                          TrainingPipelineParams)
from tests.data_generator import generate_test_dataframe
from src.pipelines.train_pipeline import (save_artifacts,
                                          evaluate_model,
                                          train_model,
                                          prepare_data,
                                          prepare_feature_transformer,
                                          transform_train_data,
                                          transform_valid_data,
                                          train)


@pytest.fixture()
def nope_logger() -> Logger:
    return NopeLogger('nope')


@pytest.fixture()
def simple_trainer(nope_logger: Logger, training_params: TrainingParams) -> \
        ModelTrainer:
    return ModelTrainer(training_params, nope_logger)


@pytest.fixture()
def medium_data() -> pd.DataFrame:
    return generate_test_dataframe(10, 9)


@pytest.fixture()
def big_data() -> pd.DataFrame:
    return generate_test_dataframe(100, 9)


@pytest.fixture()
def cat_features() -> list:
    return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']


@pytest.fixture()
def num_features() -> list:
    return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']


@pytest.fixture()
def all_features() -> list:
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


@pytest.fixture()
def feature_params(cat_features, num_features) -> FeatureParams:
    return FeatureParams(categorical_feats=cat_features,
                         numerical_feats=num_features)


@pytest.fixture()
def train_feature_builder(nope_logger, all_features, feature_params) -> \
        FeatureBuilder:
    return FeatureBuilder(all_features, feature_params, nope_logger, 'train')


@pytest.fixture()
def test_feature_builder(nope_logger, all_features, feature_params) -> \
        FeatureBuilder:
    return FeatureBuilder(all_features, feature_params, nope_logger, 'test')


@pytest.fixture()
def training_params() -> TrainingParams:
    return TrainingParams(model_fixed_params={'random_state': 9},
                          model_tune_params={'n_estimators': [1, 3, 5]})


@pytest.fixture()
def data_params(tmpdir: str) -> DataParams:
    return DataParams(
        data_path=tmpdir.join('data.csv'),
        model_path=tmpdir.join('model.pkl'),
        metric_path=tmpdir.join('metrics.json'),
        params_path=tmpdir.join('params.json'))


@pytest.fixture()
def pipeline_config_params(tmpdir: str, nope_logger: Logger,
                           data_params: DataParams,
                           feature_params: FeatureParams,
                           training_params: TrainingParams):
    app_params = AppParams(logging=None)
    train_params = TrainingPipelineParams(data_params=data_params,
                                          feature_params=feature_params,
                                          train_params=training_params)
    return ConfigTrainParams(app_params, train_params)


def test_save_artifacts(data_params: DataParams, nope_logger: Logger,
                        simple_trainer: ModelTrainer):
    scores = {
        'train_scores': {'roc_auc': 0.9, 'accuracy': 0.96},
        'valid_scores': {'roc_auc': 0.8, 'accuracy': 0.81},
    }
    simple_trainer.tune_params = {'n_estimators': 3}
    simple_trainer.model = object()

    save_artifacts(nope_logger, trainer=simple_trainer,
                   scores=scores,
                   params=data_params)

    assert os.path.exists(data_params.model_path)
    assert os.path.exists(data_params.metric_path)
    assert os.path.exists(data_params.params_path)


def test_prepare_data(tmpdir: str, medium_data: pd.DataFrame,
                      nope_logger: Logger):
    filepath = tmpdir.join('test.csv')
    medium_data.to_csv(filepath, index=False)
    data_params = DataParams(
        data_path=filepath,
        splitting_params=SplittingParams(random_state=9, val_size=0.2))
    train_df, valid_df = prepare_data(nope_logger, data_params)

    assert set(train_df.index) == {7, 2, 1, 9, 3, 0, 6, 5}
    assert set(valid_df.index) == {8, 4}


def test_prepare_feature_transformer(nope_logger: Logger,
                                     feature_params: FeatureParams,
                                     medium_data: pd.DataFrame):
    pipeline = \
        prepare_feature_transformer(nope_logger, feature_params, medium_data)
    assert pipeline is not None


def test_transform_train_data(nope_logger: Logger,
                              train_feature_builder: FeatureBuilder,
                              medium_data: pd.DataFrame):
    pipeline = train_feature_builder.transformer
    pipeline.fit(medium_data)
    data = transform_train_data(nope_logger, pipeline, medium_data)

    assert data.shape == (4, 22)
    assert np.allclose(data[:, 0].flatten(), np.array([1, 0, 1, 1]))
    assert np.allclose(data[:, 3].flatten(), np.array([0, 1, 1, 0]))
    assert np.allclose(
        data[:, -3].flatten(),
        np.array([0.68045656, 1.61366809, 0.95843267, 2.54229133]))


def test_transform_valid_data(nope_logger: Logger,
                              test_feature_builder: FeatureBuilder,
                              medium_data: pd.DataFrame):
    pipeline = test_feature_builder.transformer
    pipeline.fit(medium_data)
    data = transform_valid_data(nope_logger, pipeline, medium_data)

    assert data.shape == (10, 25)
    assert np.allclose(data[:4, 0].flatten(), np.array([1, 1, 0, 1]))
    assert np.allclose(data[:4, 3].flatten(), np.array([0, 0, 1, 0]))
    assert np.allclose(
        data[:4, -3].flatten(),
        np.array([0.68045656, 1.1189113, 1.61366809, 2.69485727]))


def test_train_model(nope_logger: Logger,
                     train_feature_builder: FeatureBuilder,
                     training_params: TrainingParams,
                     big_data: pd.DataFrame):
    pipeline = train_feature_builder.transformer
    data = pipeline.fit_transform(big_data)
    train_ft = data[:50]
    valid_ft = data[50:]
    trainer = train_model(nope_logger, training_params, train_ft, valid_ft)

    assert trainer.model is not None
    assert 0.55 < trainer.score < 0.65
    assert trainer.tune_params == {'n_estimators': 3}


def test_evaluate_model(nope_logger: Logger,
                        train_feature_builder: FeatureBuilder,
                        training_params: TrainingParams,
                        big_data: pd.DataFrame):
    pipeline = train_feature_builder.transformer
    data = pipeline.fit_transform(big_data)
    train_ft = data[:50]
    valid_ft = data[50:]
    trainer = train_model(nope_logger, training_params, train_ft, valid_ft)
    scores = evaluate_model(nope_logger, trainer, train_ft, valid_ft)

    assert 0.85 < scores['train_scores']['accuracy'] < 0.92
    assert 0.88 < scores['train_scores']['roc_auc'] < 0.92
    assert 0.87 < scores['train_scores']['f1'] < 0.92
    assert 0.89 < scores['train_scores']['precision'] < 0.93
    assert 0.85 < scores['train_scores']['recall']
    assert 0.9 < scores['valid_scores']['accuracy'] < 0.95
    assert 0.9 < scores['valid_scores']['roc_auc'] < 0.95
    assert 0.92 < scores['valid_scores']['f1'] < 0.93
    assert 0.84 < scores['valid_scores']['precision'] < 0.87
    assert 0.98 < scores['valid_scores']['recall']


def test_train(nope_logger: Logger,
               big_data: pd.DataFrame,
               pipeline_config_params: ConfigTrainParams):
    data_params = pipeline_config_params.train.data_params
    big_data.to_csv(data_params.data_path, index=False)
    train(nope_logger, pipeline_config_params)

    scores = json.load(data_params.metric_path)

    assert os.path.exists(data_params.params_path)
    assert os.path.exists(data_params.model_path)
    assert os.path.exists(data_params.metric_path)
    assert 0.82 < scores['train_scores']['accuracy'] < 0.85
    assert 0.80 < scores['train_scores']['f1'] < 0.87
    assert 0.77 < scores['train_scores']['recall']
    assert 0.9 < scores['valid_scores']['roc_auc'] <= 1.0
    assert 0.84 < scores['valid_scores']['precision'] <= 1.0
