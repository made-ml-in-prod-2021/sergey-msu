from typing import List
from dataclasses import dataclass, field


@dataclass()
class TransformParams:
    """ Data transform config ORM class. """
    quantile: float = 0.05
    feats: List[str] = field(default_factory=list)
