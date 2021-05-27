import os
import json
import logging
import pandas as pd
import click
import pickle
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


@click.command('model_train')
@click.option('--input-path')
@click.option('--output-path')
def model_train(input_path: str, output_path: str):
    log('Begin train model...')

    if not os.path.exists(input_path):
        raise ValueError(f'Data path {input_path} does not exist')

    data_train_path = os.path.join(input_path, 'data_train.csv')
    if not os.path.exists(data_train_path):
        raise ValueError(f'Train data file {data_train_path} does not exist')

    log(f'Load train data from {data_train_path}...')
    data_train = pd.read_csv(data_train_path)
    X_train = data_train.drop('target', axis=1)
    y_train = data_train['target'].values

    log(f'Grid search train model...')
    model = RandomForestClassifier(random_state=9)
    params = {'n_estimators': [10, 30, 50],
              'max_depth': [1, 5, 10]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)
    grid = GridSearchCV(model, params, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    log(f'Best model accuracy:' + str(round(grid.best_score_, 2)))
    log(f'Best model params:' + str(grid.best_params_))

    log(f'Save model and train results...')
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'model.pkl'), 'wb') as f:
        pickle.dump(grid.best_estimator_, f)
    with open(os.path.join(output_path, 'train_stats.json'), 'w') as f:
        json.dump({'accuracy': grid.best_score_}, f)

    log('Model train success')


def log(message):
    logger = logging.getLogger('model_train')
    logger.warning(message)


if __name__ == '__main__':
    model_train()
