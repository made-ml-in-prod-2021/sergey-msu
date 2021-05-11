from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    """ Model training config ORM class. """
    model_type: str = 'sklearn.ensemble.RandomForestClassifier'
    model_fixed_params: Dict[str, Any] = \
        field(default_factory=dict)
    model_tune_params: Dict[str, Any] = \
        field(default_factory=dict)
