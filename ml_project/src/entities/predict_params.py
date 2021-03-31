from dataclasses import dataclass, field
from .logger_params import LoggerParams


@dataclass()
class PredictParams:
    """ Model predictions config ORM class. """
    name: str = 'predict'
    experiment: str = 'default_experiment'
    data_path: str = 'heart.csv'
    model_path: str = 'model.pkl'
    save_path: str = 'results.csv'
    logging: LoggerParams = field(default_factory=LoggerParams)
