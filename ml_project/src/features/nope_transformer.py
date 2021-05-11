import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class NopeTransformer(BaseEstimator, TransformerMixin):
    """ Dummy class to do pass-through dataset as is. """
    def __init__(self, cols: List[str] = None, quantile: float = 0.05):
        self.cols = cols
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y: np.array = None) -> pd.DataFrame:
        return self

    def transform(self, X: pd.DataFrame, y: np.array = None) -> pd.DataFrame:
        return X
