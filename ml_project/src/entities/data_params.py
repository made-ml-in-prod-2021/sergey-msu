from typing import Optional
from dataclasses import dataclass, field

from .split_params import SplittingParams


@dataclass()
class DataParams:
    """ Data config ORM class. """
    data_path: str = 'heart.csv'
    model_path: str = 'model.pkl'
    metric_path: str = 'metrics.json'
    params_path: str = 'params.json'
    preds_path: Optional[str] = 'preds.csv'
    splitting_params: SplittingParams = field(default_factory=SplittingParams)
