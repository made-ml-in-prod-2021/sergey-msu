from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from ..entities.split_params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    """
    Read csv data from file.

    Parameters
    ----------
    path: str
        Path to csv datafile.

    """
    data = pd.read_csv(path)
    return data


def split_train_valid_data(data: pd.DataFrame, params: SplittingParams) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test chunks.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to split.
    params: SplittingParams
        Splitting params e.g.test size etc.

    """
    train_data, valid_data = \
        train_test_split(data,
                         test_size=params.val_size,
                         random_state=params.random_state)

    return train_data, valid_data
