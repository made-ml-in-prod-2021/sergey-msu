import os
import hydra
import logging
from logging import Logger
from collections import Counter
import pandas as pd

from src.utils import create_logger
from src.entities import EDAReportParams, LoggerParams


CAT_FEATS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
NUM_FEATS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
TARGET_COl = 'target'


def setup_pandas():
    """ Setup pandas to visualiza all data. """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)


def create_report(writer: Logger, config_params: EDAReportParams):
    """
    Create EDA report.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    config_params: EDAReportParams
        All report configuration parameters.

    """
    data = pd.read_csv(config_params.data_path)

    print_header(writer, config_params.data_path)
    print_stats(writer, data)
    print_distributions(writer, data)
    print_correlations(writer, data)


def print_header(writer: Logger, data_path: str):
    """
    Create EDA report.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    data_path: str
        Path to data.

    """
    title = ' EDA - Heart Disease UCI '
    writer.info('*'*(40 + len(title)))
    writer.info('*'*20 + title + '*'*20)
    writer.info('*'*(40 + len(title)))
    writer.info(f'\nData source: {os.path.abspath(data_path)}\n')


def print_stats(writer: Logger, data: pd.DataFrame):
    """
    Create EDA report.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    data: pd.DataFrame
        Dataframe.

    """
    writer.info(f'*** Data shape:\n{data.shape}\n')
    writer.info(f'*** Head:\n{data.head()}\n')
    writer.info(f'*** Describe:\n{data.describe().round(2)}\n')
    writer.info(f'*** Missing:\n{len(data) - len(data.dropna())}\n')
    writer.info(f'*** Targets:\n{dict(Counter(data[TARGET_COl]))}\n')


def print_distributions(writer: Logger, data: pd.DataFrame):
    """
    Print distribution analysis results.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    data: pd.DataFrame
        Dataframe.

    """
    writer.info('*** Categorical Distributions:')
    max_len = max(map(len, CAT_FEATS))
    for feat in CAT_FEATS:
        writer.info(f'  {feat.ljust(max_len)}: {dict(Counter(data[feat]))}')
    writer.info('')


def print_correlations(writer: Logger, data: pd.DataFrame):
    """
    Print correlation analysis results.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    data: pd.DataFrame
        Dataframe.

    """
    writer.info('*** Feature Correlations:')
    writer.info(data.corr().round(2))


@hydra.main(config_path='../../configs/ext', config_name='eda_report')
def main(params: EDAReportParams):
    """
    Pipeline entry point.

    Parameters
    ----------
    params: EDAReportParams
        All report configuration parameters.

    """
    writer_params = LoggerParams(path=params.save_path,
                                 format='%(message)s',
                                 date_format='%Y-%m-%d %H:%M:%S',
                                 stdout=False,
                                 level=logging.INFO,
                                 mode='w+')
    writer = create_logger(params.name, writer_params)

    setup_pandas()
    create_report(writer, params)


if __name__ == '__main__':
    main()
