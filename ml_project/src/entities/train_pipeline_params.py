from dataclasses import dataclass, field
from typing import Optional, List
from .data_params import DataParams
from .feature_params import FeatureParams
from .train_params import TrainingParams


@dataclass()
class PipelineParams:
    """ Base pipeline config ORM class. """
    defaults: List[str] = field(default_factory=list)
    data_params: Optional[DataParams] = None
    feature_params: Optional[FeatureParams] = None


@dataclass()
class TrainingPipelineParams(PipelineParams):
    """ Training pipeline config ORM class. """
    train_params: Optional[TrainingParams] = None


@dataclass()
class PredictionPipelineParams(PipelineParams):
    """ Predictions pipeline config ORM class. """
    pass
