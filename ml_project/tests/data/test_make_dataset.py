import pytest
import numpy as np
import pandas as pd

from src.data.make_dataset import read_data, split_train_valid_data
from src.entities import SplittingParams
from tests.data_generator import generate_test_dataframe


@pytest.fixture()
def small_data() -> pd.DataFrame:
    return generate_test_dataframe(3, 9)


@pytest.fixture()
def medium_data() -> pd.DataFrame:
    return generate_test_dataframe(10, 9)


def test_read_data(tmpdir, small_data):
    filepath = tmpdir.join('test.csv')
    small_data.to_csv(filepath, index=False)
    data = read_data(filepath)

    assert np.allclose(small_data.values, data.values)


def test_split_train_valid_data(medium_data):
    splitting_params = SplittingParams(random_state=9, val_size=0.2)
    train, valid = split_train_valid_data(medium_data, splitting_params)

    assert set(train.index) == {7, 2, 1, 9, 3, 0, 6, 5}
    assert set(valid.index) == {8, 4}
