from typing import Optional
from dataclasses import dataclass, field

from .logger_params import LoggerParams


@dataclass()
class AppParams:
    """ Application config ORM class. """
    name: str = 'default_project'
    experiment: str = 'default_experiment'
    logging: Optional[LoggerParams] = field(default_factory=LoggerParams)
