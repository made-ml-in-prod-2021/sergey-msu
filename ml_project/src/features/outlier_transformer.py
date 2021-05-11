import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that simply filters data that outlie
    quantile bound.

    """
    def __init__(self, cols: List[str] = None, quantile: float = 0.05):
        self.cols = cols
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y: np.array = None):
        """ Dummy fit. """
        return self

    def transform(self, X: pd.DataFrame, y: np.array = None) -> pd.DataFrame:
        """ Quantile filtering transform. """
        X = X.copy()
        cols = self.cols or X.columns.values
        cols = list(set(cols).intersection(X.columns.values))
        stats = {}

        for col in cols:
            min_value = X[col].quantile(self.quantile)
            max_value = X[col].quantile(1 - self.quantile)
            stats[col] = (min_value, max_value)

        for col in cols:
            min_value, max_value = stats[col]
            X = X[(X[col] >= min_value) & (X[col] <= max_value)]

        return X
