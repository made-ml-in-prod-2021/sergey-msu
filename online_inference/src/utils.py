""" Application utility functions. """

import os
import random
import logging
from logging import Logger
import numpy as np

from src.entities.logger_params import LoggerParams


def create_logger(name: str, params: LoggerParams) -> Logger:
    """
    Create application logger.

    Parameters
    ----------
    name: str
        Logger name.
    params: LoggerParams
        Logger configuration params.

    """
    folder = os.path.dirname(params.path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    logger = logging.getLogger(name)
    logger.setLevel(params.level)

    simple_formatter = logging.Formatter(fmt=params.format,
                                         datefmt=params.date_format)

    # file handler
    file_handler = logging.FileHandler(filename=params.path, mode=params.mode)
    file_handler.setLevel(params.level)
    file_handler.setFormatter(simple_formatter)
    logger.addHandler(file_handler)

    # console handler
    if params.stdout:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(params.level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str, params: LoggerParams = None, ) -> Logger:
    """
    Get or create application logger.

    Parameters
    ----------
    name: str
        Logger name.
    params: LoggerParams
        Logger configuration params.

    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return create_logger(name, params)
    return logger


def set_random(seed):
    """
    Fix random at some seed.

    Parameters
    ----------
    seed: int
        Seed to set random with (None for randomness).

    """
    random.seed(seed)
    np.random.seed(seed)
