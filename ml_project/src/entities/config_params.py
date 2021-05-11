from dataclasses import dataclass, field

from .app_params import AppParams
from .train_pipeline_params import (TrainingPipelineParams,
                                    PredictionPipelineParams)


@dataclass()
class ConfigParams:
    """ Config base config ORM class. """
    app: AppParams = field(default_factory=AppParams)


@dataclass()
class ConfigTrainParams(ConfigParams):
    """ Train config ORM class. """
    train: TrainingPipelineParams = \
        field(default_factory=TrainingPipelineParams)


@dataclass()
class ConfigPredParams(ConfigParams):
    """ Prediction config ORM class. """
    pred: PredictionPipelineParams = \
        field(default_factory=PredictionPipelineParams)
