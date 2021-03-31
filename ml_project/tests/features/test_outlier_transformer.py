import pytest
import numpy as np
import pandas as pd

from src.features.outlier_transformer import OutlierTransformer
from tests.data_generator import generate_test_dataframe


@pytest.fixture()
def med_data_age() -> pd.DataFrame:
    return generate_test_dataframe(10, 9, cols_data={'age': list(range(10))})


@pytest.fixture()
def med_data_age_chol() -> pd.DataFrame:
    return generate_test_dataframe(50, 9,
                                   cols_data={'age': list(range(50)),
                                              'chol': list(range(250, 300))})


def test_outlier_transform_init():
    transformer = OutlierTransformer(cols=['a', 'b'], quantile=0.0125)
    assert transformer.cols == ['a', 'b']
    assert transformer.quantile == 0.0125


@pytest.mark.parametrize(
    'quantile',
    [
        pytest.param(0.05),
        pytest.param(0.1),
        pytest.param(0.15),
        pytest.param(0.2),
        pytest.param(0.25),
        pytest.param(0.45)
    ]
)
def test_outlier_transform_one_column_quantiles(med_data_age, quantile):
    min_value = med_data_age['age'].quantile(quantile)
    max_value = med_data_age['age'].quantile(1 - quantile)
    truth = med_data_age[(med_data_age['age'] >= min_value) &
                         (med_data_age['age'] <= max_value)]

    transformer = OutlierTransformer(cols=['age'], quantile=quantile)
    result = transformer.transform(med_data_age)

    assert np.allclose(truth.values, result.values)


@pytest.mark.parametrize(
    'quantile',
    [
        pytest.param(0.05),
        pytest.param(0.1),
        pytest.param(0.15),
        pytest.param(0.2),
        pytest.param(0.25),
        pytest.param(0.45)
    ]
)
def test_outlier_transform_two_columns_quantiles(med_data_age_chol, quantile):
    truth = med_data_age_chol.copy()
    for col in ['age', 'chol']:
        min_value = med_data_age_chol[col].quantile(quantile)
        max_value = med_data_age_chol[col].quantile(1 - quantile)
        truth = truth[(truth[col] >= min_value) & (truth[col] <= max_value)]

    transformer = OutlierTransformer(cols=['age', 'chol'], quantile=quantile)
    result = transformer.transform(med_data_age_chol)

    assert np.allclose(truth.values, result.values)
