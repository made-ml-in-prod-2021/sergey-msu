from dataclasses import dataclass, field
from typing import List, Optional

from .transform_params import TransformParams


@dataclass()
class FeatureParams:
    """ Feature extraction config ORM class. """
    cat_missing: str = 'most_frequent'
    num_missing: str = 'mean'
    transform_params: TransformParams = field(default_factory=TransformParams)
    categorical_feats: List[str] = field(default_factory=list)
    numerical_feats: List[str] = field(default_factory=list)
    target_col: Optional[str] = None
