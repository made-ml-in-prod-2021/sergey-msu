import numpy as np
from logging import Logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.entities.feature_params import FeatureParams
from .outlier_transformer import OutlierTransformer
from .nope_transformer import NopeTransformer


class FeatureBuilder:
    """ Class that incapsulates feature building logic. """
    def __init__(self, all_features: list, params: FeatureParams,
                 logger: Logger,
                 mode: str = 'train'):
        """
        Init class instance.

        Parameters
        ----------
        all_features: list
            List of features under interest.
        params: FeatureParams
            Feature pipeline parameters.
        logger: Logger
            App Logger.
        mode: str
            Feature pipeline mode (i.e. 'train' or anything else treating as
            test mode).

        """
        self.mode = mode
        self.all_features = all_features
        self.params = params
        self.logger = logger

        self.logger.info('  > build feature transformer')
        self.transformer = self.build(params)
        self.logger.info('  > build feature transformer: done')

    def build(self, params: FeatureParams) -> Pipeline:
        """
        Build feature pipeline.

        Parameters
        ----------
        params: FeatureParams
            Feature pipeline parameters.

        """
        cat_feats = list(params.categorical_feats)
        num_feats = list(params.numerical_feats)
        app_feats = [f for f in self.all_features
                     if (f not in cat_feats) and (f not in num_feats)]

        transformer = Pipeline(
            [
                (
                    'row_pipeline',
                    self.build_filter(params),
                ),
                (
                    'column_pipeline',
                    ColumnTransformer(
                        [
                            (
                                'categorical_pipeline',
                                self.build_categorical(params),
                                cat_feats,
                            ),
                            (
                                'numerical_pipeline',
                                self.build_numerical(params),
                                num_feats,
                            ),
                            (
                                'column_append',
                                NopeTransformer(),
                                app_feats,
                            ),
                        ])
                )
            ])
        return transformer

    def build_filter(self, params: FeatureParams) -> Pipeline:
        """
        Build initial row filter.

        Parameters
        ----------
        params: FeatureParams
            Feature pipeline parameters.

        """
        self.logger.info('    > build filter')
        if self.mode == 'train':
            params = params.transform_params
            result = Pipeline(
                [
                    ('quantile', OutlierTransformer(cols=params.feats,
                                                    quantile=params.quantile))
                ]
            )
        else:
            result = Pipeline([('nope', NopeTransformer())])

        self.logger.info('    > build filter: done')
        return result

    def build_categorical(self, params: FeatureParams) -> Pipeline:
        """
        Build categorical feature pipeline.

        Parameters
        ----------
        params: FeatureParams
            Feature pipeline parameters.

        """
        self.logger.info('    > build categorical')
        if self.mode == 'train':
            result = Pipeline(
                [
                    ('impute', SimpleImputer(missing_values=np.nan,
                                             strategy=params.cat_missing)),
                    ('ohe', OneHotEncoder()),
                ]
            )
        else:
            result = Pipeline([('ohe', OneHotEncoder())])

        self.logger.info('    > build categorical:done')
        return result

    def build_numerical(self, params: FeatureParams) -> Pipeline:
        """
        Build numerical feature pipeline.

        Parameters
        ----------
        params: FeatureParams
            Feature pipeline parameters.

        """
        self.logger.info('    > build numerical')
        if self.mode == 'train':
            result = Pipeline(
                [
                    ('impute', SimpleImputer(missing_values=np.nan,
                                             strategy=params.num_missing))
                ]
            )
        else:
            result = Pipeline([('nope', NopeTransformer())])
        self.logger.info('    > build numerical: done')
        return result
