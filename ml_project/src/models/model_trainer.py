from typing import Union, Dict
from logging import Logger
from pydoc import locate
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             f1_score,
                             precision_score,
                             recall_score)

from src.entities.train_params import TrainingParams


class ModelTrainer:
    """ Class that incapsulates model training logic. """
    def __init__(self, params: TrainingParams, logger: Logger):
        """
        Init class instance.

        Parameters
        ----------
        params: TrainingParams
            Model training pipeline parameters.
        logger: Logger
            App logger.

        """
        self.model = None
        self.score = None
        self.tune_params = None
        self.params = params
        self.fixed_params = dict(params.model_fixed_params)
        self.logger = logger

        self.logger.info('  > create model blueprint')
        model_type = locate(params.model_type)
        self.model_blueprint = model_type(**self.fixed_params)
        self.logger.info('  > create model blueprint: done')

    def fit(self, train_ft: np.array, valid_ft: np.array):
        """
        Fit underlying model.

        Parameters
        ----------
        train_ft: np.array
            Train dataset.
        valid_ft: np.array
            Valid dataset.
        params: TrainingParams
            Model training pipeline parameters.

        """
        self.logger.info('    > begin fit model')
        x_train = train_ft[:, :-1]
        y_train = train_ft[:, -1]
        x_valid = valid_ft[:, :-1]
        y_valid = valid_ft[:, -1]

        x = np.concatenate((x_train, x_valid))
        y = np.concatenate((y_train, y_valid))

        split_index = [-1] * len(x_train) + [0] * len(x_valid)
        pds = PredefinedSplit(test_fold=split_index)
        self.logger.info('    > predefined split: created')

        self.logger.info('    > begin grid search')
        param_grid = dict(self.params.model_tune_params)
        clf = GridSearchCV(estimator=self.model_blueprint,
                           cv=pds,
                           scoring='accuracy',
                           param_grid=param_grid)
        clf.fit(x, y)
        self.logger.info('    > grid search: done')

        self.model = clf.best_estimator_
        self.score = clf.best_score_
        self.tune_params = clf.best_params_
        self.logger.info('    > model fit: done')

    def predict(self, X: Union[np.array, pd.DataFrame]):
        """
        Do model predict.

        Parameters
        ----------
        X: np.array or pd.DataSet
            Datase to predict result.

        """
        if self.model is None:
            raise ValueError('train model before using')

        self.logger.info('    > model prediction')
        result = self.model.predict(X)
        self.logger.info('    > model prediction: done')

        return result

    def evaluate(self, X: Union[np.array, pd.DataFrame], y: np.array) \
            -> Dict[str, int]:
        """
        Do model evaluate.

        Parameters
        ----------
        X: np.array or pd.DataSet
            Datase to predict result.
        y: np.array
            Ground truth.

        """
        self.logger.info('    > model evaluation')
        y_pred = self.predict(X)
        self.logger.info('    > model evaluation: done')

        return {
            'accuracy': round(accuracy_score(y, y_pred), 4),
            'roc_auc': round(roc_auc_score(y, y_pred), 4),
            'f1': round(f1_score(y, y_pred), 4),
            'precision': round(precision_score(y, y_pred), 4),
            'recall': round(recall_score(y, y_pred), 4)
        }
