import pytest
import pandas as pd

from src.features.nope_transformer import NopeTransformer
from tests.data_generator import generate_test_dataframe


@pytest.fixture()
def medium_data() -> pd.DataFrame:
    return generate_test_dataframe(10, 9)


def test_nope_transformer_transform(medium_data):
    transformer = NopeTransformer()
    data = transformer.transform(medium_data)

    assert data.equals(medium_data)
