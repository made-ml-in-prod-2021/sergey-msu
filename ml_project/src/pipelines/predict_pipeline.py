""" Main model predict pipeline. """

import os
import hydra
import pickle
from logging import Logger
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from src.data.make_dataset import read_data
from src.entities import ConfigPredParams, DataParams, FeatureParams
from src.features.feature_builder import FeatureBuilder
from src.utils import create_logger


def predict(logger: Logger, params: ConfigPredParams):
    """
    Run main prediction pipeline.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: ConfigPredParams
        All configuration parameters.

    """
    logger.info('=' * 80)
    logger.info(f'begin predict experiment: {params.app.experiment}')
    logger.info('=' * 80)

    try:
        logger.info(f'start prediction pipeline with params {params}')

        # load data
        data_df = prepare_data(logger, params.pred.data_params)
        params = params.pred

        # extract features
        transformer = prepare_feature_transformer(logger,
                                                  params.feature_params,
                                                  data_df)
        data_ft = transform_pred_data(logger, transformer, data_df)

        # load model
        model = load_model(logger, params.data_params)

        # do predictions
        preds = predict_model(logger, model, data_ft)

        # # save predictions
        save_predictions(logger, preds, params.data_params)

        logger.info('done!')

    except Exception as ex:
        logger.exception(ex)


def save_predictions(logger: Logger, preds: np.array, params: DataParams):
    """
    Save predictions file to drive.

    Parameters
    ----------
    logger: Logger
        Application logger.
    preds: np.array
        Prediction array.
    params: DataParams
        Data parameters.

    """
    logger.info(f'save predicstions to  {os.path.abspath(params.preds_path)}')
    preds_dir = os.path.dirname(params.preds_path)
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
    pd.DataFrame({'preds': preds}).to_csv(params.preds_path)


def prepare_data(logger: Logger, params: DataParams) -> pd.DataFrame:
    """
    Read test data from drive.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: DataParams
        Data parameters.

    """
    data_df = read_data(params.data_path)
    if 'target' in data_df.columns:
        data_df.drop('target', axis=1, inplace=True)
    logger.info(f'data_df.shape is  {data_df.shape}')
    return data_df


def prepare_feature_transformer(logger: Logger,
                                params: FeatureParams,
                                data: pd.DataFrame) -> Pipeline:
    """
    Build data-to-feature transformer.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: FeatureParams
        Feature construct parameters.
    data: pd.DataFrame
        Raw data.

    """
    logger.info(f'build transformer with params {params}')
    builder = FeatureBuilder(data.columns.values, params, mode='predict')
    return builder.transformer


def transform_pred_data(logger: Logger,
                        transformer: Pipeline,
                        data: pd.DataFrame) -> np.array:
    """
    Build data-to-feature transformer.

    Parameters
    ----------
    logger: Logger
        Application logger.
    transformer: Pipeline
        Data-to-feature transformer.
    data: pd.DataFrame
        Raw data.

    """
    logger.info(f'transform data: before shape is {data.shape}')
    transformer.fit(data)
    data_feats = transformer.transform(data)
    logger.info(f'transform data: after shape is  {data_feats.shape}')

    return data_feats


def load_model(logger: Logger, params: DataParams) -> BaseEstimator:
    """
    Load trained model from file.

    Parameters
    ----------
    logger: Logger
        Application logger.
    params: DataParams
        Data parameters.

    """
    logger.info('loading model from file: '
                f'{os.path.abspath(params.model_path)}')
    with open(params.model_path, 'rb') as file:
        model = pickle.load(file)
        logger.info(f'    model loaded: {type(model)}')
    return model


def predict_model(logger: Logger, model: BaseEstimator, data: np.array) \
        -> np.array:
    """
    Load trained model from file.

    Parameters
    ----------
    logger: Logger
        Application logger.
    model: BaseEstimator
        Trained model.
    data: np.array
        Features data.

    """
    logger.info('do model predictions')
    preds = model.predict(data)
    logger.info(f'predictions done: {len(preds)}')
    return preds


@hydra.main(config_path='../../configs', config_name='config_pred')
def main(params: ConfigPredParams):
    """
    Pipeline entry point.

    Parameters
    ----------
    params: ConfigPredParams
        All configuration parameters.

    """
    # read application configuration
    params.app.logging.path = \
        params.app.logging.path.format(experiment=params.app.experiment)

    # create application logger
    logger = create_logger(params.app.name, params.app.logging)

    # run training pipeline
    predict(logger, params)


if __name__ == '__main__':
    main()
