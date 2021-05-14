import yaml
from typing import Optional
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from .logger_params import LoggerParams


@dataclass()
class RequestParams:
    """ Application config ORM class. """
    host: str = '0.0.0.0'
    port: int = 8000
    test_data_path: str = '../data/test_data.csv'
    logging: Optional[LoggerParams] = field(default_factory=LoggerParams)


def read_req_params(path: str) -> RequestParams:
    """ Read test request parameters from config file. """
    with open(path, 'r') as input_stream:
        app_schema = class_schema(RequestParams)()
        app_params = app_schema.load(yaml.safe_load(input_stream))
        return app_params
