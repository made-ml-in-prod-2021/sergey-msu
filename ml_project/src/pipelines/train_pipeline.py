import os
import json
import pickle
from logging import Logger
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.pipeline import Pipeline
import hydra

from src.data.make_dataset import read_data, split_train_valid_data
from src.features.feature_builder import FeatureBuilder
from src.models.model_trainer import ModelTrainer
from src.utils import create_logger
from src.entities import (ConfigTrainParams,
                          TrainingParams,
                          DataParams,
                          FeatureParams)


def train(logger: Logger, params: ConfigTrainParams):
    """
    Run main training pipeline.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: ConfigTrainParams
        All configuration parameters.

    """
    logger.info('=' * 80)
    logger.info(f'begin train experiment: {params.app.experiment}')
    logger.info('=' * 80)

    try:
        logger.info(f'start train pipeline with params {params}')
        params = params.train

        # load data
        train_df, valid_df = prepare_data(logger, params.data_params)

        # extract features
        transformer = prepare_feature_transformer(logger,
                                                  params.feature_params,
                                                  train_df)
        train_ft = transform_train_data(logger, transformer, train_df)
        valid_ft = transform_valid_data(logger, transformer, valid_df)

        # train grid-cv model
        trainer = train_model(logger, params.train_params, train_ft, valid_ft)

        # evaluate model
        scores = evaluate_model(logger, trainer, train_ft, valid_ft)

        # save model, metrics and best params
        save_artifacts(logger, trainer, scores, params.data_params)

        logger.info('done!')

    except Exception as ex:
        logger.exception(ex)


def save_artifacts(logger: Logger, trainer: ModelTrainer,
                   scores: Dict[str, Dict[str, float]],
                   params: DataParams):
    """
    Save all training artifacts to drive.

    Parameters
    ----------
    logger: Logger
        Application logger.
    trainer: ModelTrainer
        Model trainer.
    scores: Dict[str, Dict[str, float]]
        Evaluated model metrics.
    params: DataParams
        Data parameters.

    """
    logger.info('save training artifacts:')

    with open(params.metric_path, 'w') as file:
        json.dump(scores, file)
        logger.info('metrics saved to '
                    f'{os.path.abspath(params.metric_path)}')

    with open(params.params_path, 'w') as file:
        model_params = trainer.fixed_params.copy()
        model_params.update(trainer.tune_params)
        json.dump(model_params, file)
        logger.info('best model params saved to '
                    f'{os.path.abspath(params.params_path)}')

    with open(params.model_path, 'wb') as file:
        pickle.dump(trainer.model, file)
        logger.info('model saved to '
                    f'{os.path.abspath(params.model_path)}')


def evaluate_model(logger: Logger, trainer: ModelTrainer,
                   train_ft: np.array, valid_ft: np.array) \
        -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on tran and valid datasets.

    Parameters
    ----------
    logger: Logger
        Application logger.
    trainer: ModelTrainer
        Model trainer.
    train_ft: np.array
        Train feature data.
    valid_ft: np.array
        Validation feature data.

    """
    logger.info('evaluate trained model:')

    X_train = train_ft[:, :-1]
    y_train = train_ft[:, -1]
    train_scores = trainer.evaluate(X_train, y_train)
    logger.info(f'    on train data: {train_scores}')

    X_valid = valid_ft[:, :-1]
    y_valid = valid_ft[:, -1]
    valid_scores = trainer.evaluate(X_valid, y_valid)
    logger.info(f'    on valid data: {valid_scores}')

    return {'train_scores': train_scores, 'valid_scores': valid_scores}


def train_model(logger: Logger, params: TrainingParams,
                train_ft: np.array, valid_ft: np.array):
    """
    Build trainer and train model.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: TrainingParams
        All train parameters.
    train_ft: np.array
        Train feature data.
    valid_ft: np.array
        Validation feature data.

    """
    logger.info(f'begin training model {params.model_type}')
    trainer = ModelTrainer(params, logger)
    trainer.fit(train_ft, valid_ft)

    logger.info('training finished')
    logger.info(f'best score:  {trainer.score}')
    logger.info(f'best params: {trainer.tune_params}')

    return trainer


def prepare_data(logger: Logger, params: DataParams) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split all data into train and validation parts.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: DataParams
        All data parameters.

    """
    data_df = read_data(params.data_path)
    logger.info(f'data_df.shape is  {data_df.shape}')

    train_df, valid_df = \
        split_train_valid_data(data_df, params.splitting_params)
    logger.info(f'train_df.shape is {train_df.shape}')
    logger.info(f'valid_df.shape is {valid_df.shape}')

    return train_df, valid_df


def prepare_feature_transformer(logger: Logger,
                                params: FeatureParams,
                                data: pd.DataFrame) \
        -> Pipeline:
    """
    Build data-to-feature transformer.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: FeatureParams
        All feature parameters.
    data: pd.DataFrame
        Data to transform.

    """
    logger.info(f'build transformer with params {params}')
    builder = FeatureBuilder(data.columns.values, params, mode='train',
                             logger=logger)
    return builder.transformer


def transform_train_data(logger: Logger,
                         transformer: Pipeline,
                         data: pd.DataFrame) -> np.array:
    """
    Transform train data to feature.

    Parameters
    ----------
    logger: Logger
        Application logger.
    transformer: Pipeline
        Data-to-features transformer.
    data: pd.DataFrame
        Data to transform.

    """
    logger.info(f'transform train data: before shape is {data.shape}')
    transformer.fit(data)
    data_feats = transformer.transform(data)
    logger.info(f'transform train data: after shape is  {data_feats.shape}')

    return data_feats


def transform_valid_data(logger: Logger,
                         transformer: Pipeline,
                         data: pd.DataFrame) -> np.array:
    """
    Transform validation data to feature.

    Parameters
    ----------
    logger: Logger
        Application logger.
    transformer: Pipeline
        Data-to-features transformer.
    data: pd.DataFrame
        Data to transform.

    """
    logger.info(f'transform valid data: before shape is {data.shape}')
    data_feats = transformer.transform(data)
    logger.info(f'transform valid data: after shape is  {data_feats.shape}')

    return data_feats


@hydra.main(config_path='../../configs', config_name='config_train')
def main(params: ConfigTrainParams):
    """
    Pipeline entry point.

    Parameters
    ----------
    params: ConfigTrainParams
        All configuration parameters.

    """
    # read application configuration
    params.app.logging.path = \
        params.app.logging.path.format(experiment=params.app.experiment)

    # create application logger
    logger = create_logger(params.app.name, params.app.logging)

    # run training pipeline
    train(logger, params)


if __name__ == '__main__':
    main()
