""" Service core logic. """

import os
import pickle
import pandas as pd
from logging import Logger

from src.entities import AppParams


class ServiceCore:
    """ Service core logic class """
    def __init__(self):
        """ Init class instance. """
        self.logger = None
        self.model = None
        self.is_init = False

    def init(self, params: AppParams, logger: Logger):
        """
        Init inner service entities.

        Parameters
        ----------
        params: AppParams
            Application top-level parameters.
        logger: Logger
            Application logger

        """
        # load model
        model_path = os.path.abspath(params.model_path)
        if not os.path.exists(model_path):
            raise ValueError(f'model does not exists: {model_path}')
        logger.info(f'loading model from {model_path} ...')

        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        logger.info('model is ready')

        self.logger = logger
        self.model = model
        self.is_init = True

        self.logger.info('service core initialized successfully')

    def predict(self, data: pd.DataFrame) -> list:
        """
        Main predict logic.

        Parameters
        ----------
        data: pd.DataFrame
            Data to predict result.

        """
        datapipe = self.model['datapipe']
        classifier = self.model['model']
        data_ft = datapipe.transform(data)
        preds = classifier.predict(data_ft)

        return list(preds)
