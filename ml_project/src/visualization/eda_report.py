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
    summary = {}

    print_header(writer, config_params.data_path)
    print_stats(writer, data, summary)
    print_distributions(writer, data, summary)
    print_correlations(writer, data, summary)
    print_summary(writer, summary)


def print_summary(writer: Logger, summary: dict):
    title = ' Summary '
    writer.info('*'*(40 + len(title)))
    writer.info('*'*20 + title + '*'*20)
    writer.info('*'*(40 + len(title)))

    writer.info('Statistics:')
    writer.info(summary['stats'])
    writer.info('Disributions:')
    writer.info(summary['distr'])
    writer.info('Correlations:')
    writer.info(summary['corrs'])


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


def print_stats(writer: Logger, data: pd.DataFrame, summary: dict):
    """
    Create EDA report.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    data: pd.DataFrame
        Dataframe.
    summary: dict
        Stats aggregation

    """
    desc = data.describe().T
    shape = data.shape
    high_var = list(desc[desc['mean'] < desc['std']].index.values)
    missing = len(data) - len(data.dropna())
    traget_distr = dict(Counter(data[TARGET_COl]))

    summary['stats'] = {
        'shape': shape,
        'high_var': high_var,
        'missing': missing,
        'trarget_distr': traget_distr
    }

    writer.info(f'*** Data shape:\n{shape}\n')
    writer.info(f'*** Head:\n{data.head()}\n')
    writer.info(f'*** Describe:\n{data.describe().round(2)}\n')
    writer.info(f'*** Missing:\n{missing}\n')
    writer.info(f'*** Targets:\n{traget_distr}\n')


def print_distributions(writer: Logger, data: pd.DataFrame, summary: dict):
    """
    Print distribution analysis results.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    data: pd.DataFrame
        Dataframe.
    summary: dict
        Stats aggregation

    """
    writer.info('*** Categorical Distributions:')
    max_len = max(map(len, CAT_FEATS))
    for feat in CAT_FEATS:
        writer.info(f'  {feat.ljust(max_len)}: {dict(Counter(data[feat]))}')
    writer.info('')

    summary['distr'] = {'norm': NUM_FEATS[:3], 'lognorm': NUM_FEATS[3]}


def print_correlations(writer: Logger, data: pd.DataFrame, summary: dict):
    """
    Print correlation analysis results.

    Parameters
    ----------
    writer: Logger
        Logger that acts as file writer.
    data: pd.DataFrame
        Dataframe.
    summary: dict
        Stats aggregation

    """
    corr = data.corr().round(2)

    summary['corrs'] = {'age': None, 'cp': None, 'slope': None}
    for f in summary['corrs']:
        c = corr[f]
        c = c[abs(c) > 0.27]
        summary['corrs'][f] = list(set(c.index) - set([f]))

    writer.info('*** Feature Correlations:')
    writer.info(corr)


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
    import os
    print('>>>>', os.path.abspath(params.save_path))
    writer = create_logger(params.name, writer_params)

    setup_pandas()
    create_report(writer, params)


if __name__ == '__main__':
    main()
