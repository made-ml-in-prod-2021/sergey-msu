import pytest
import pandas as pd
from logging import Logger
from sklearn.ensemble import RandomForestClassifier

from src.models.model_trainer import ModelTrainer
from src.entities.train_params import TrainingParams
from src.features.feature_builder import FeatureBuilder, FeatureParams
from tests.data_generator import generate_test_dataframe
from tests.mocks import NopeLogger


@pytest.fixture()
def nope_logger() -> Logger:
    return NopeLogger('nope')


@pytest.fixture()
def feature_params() -> FeatureParams:
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    return FeatureParams(categorical_feats=cat_features,
                         numerical_feats=num_features)


@pytest.fixture()
def train_feature_builder(nope_logger: Logger, feature_params: FeatureParams) \
        -> FeatureBuilder:
    all_features = \
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    return FeatureBuilder(all_features, feature_params,
                          nope_logger, 'train')


@pytest.fixture()
def small_data() -> list:
    return generate_test_dataframe(100, 9)


@pytest.fixture()
def cat_features() -> list:
    return ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']


@pytest.fixture()
def num_features() -> list:
    return ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']


@pytest.fixture()
def training_params() -> TrainingParams:
    return TrainingParams(model_fixed_params={'random_state': 9},
                          model_tune_params={'n_estimators': [1, 3, 5]})


@pytest.fixture()
def simple_trainer(nope_logger: Logger, training_params: TrainingParams) -> \
        ModelTrainer:
    return ModelTrainer(training_params, nope_logger)


def test_model_trainer_init(nope_logger: Logger,
                            training_params: TrainingParams):
    trainer = ModelTrainer(training_params, nope_logger)

    assert trainer.fixed_params == {'random_state': 9}
    assert trainer.params == training_params
    assert isinstance(trainer.model_blueprint, RandomForestClassifier)
    assert trainer.model_blueprint.random_state == 9


def test_model_trainer_fit(simple_trainer: ModelTrainer,
                           train_feature_builder: FeatureBuilder,
                           small_data: pd.DataFrame):
    data = train_feature_builder.transformer.fit_transform(small_data)
    train_ft = data[:50, :]
    valid_ft = data[50:, :]
    simple_trainer.fit(train_ft, valid_ft)

    assert simple_trainer.model is not None
    assert 0.55 < simple_trainer.score < 0.65
    assert simple_trainer.tune_params == {'n_estimators': 3}


def test_model_trainer_predict(simple_trainer: ModelTrainer,
                               train_feature_builder: FeatureBuilder,
                               small_data: pd.DataFrame):
    data = train_feature_builder.transformer.fit_transform(small_data)
    simple_trainer.fit(data[:50, :], data[50:, :])
    preds = simple_trainer.predict(data[-3:, :-1])
    assert list(preds) == [0, 1, 0]


def test_model_trainer_evaluate(simple_trainer: ModelTrainer,
                                train_feature_builder: FeatureBuilder,
                                small_data: pd.DataFrame):
    data = train_feature_builder.transformer.fit_transform(small_data)
    simple_trainer.fit(data[:50, :], data[50:, :])
    scores = simple_trainer.evaluate(data[50:, :-1], data[50:, -1])

    assert 0.9 < scores['accuracy'] < 0.95
    assert 0.9 < scores['roc_auc'] < 0.95
    assert 0.92 < scores['f1'] < 0.93
    assert 0.84 < scores['precision'] < 0.87
    assert 0.98 < scores['recall']
