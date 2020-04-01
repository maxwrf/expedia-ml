import pandas as pd
from source.data.data import Data
from source.data.features import Features
from source.models.decision_tree import DecisionTree
from source.models.random_forest import RandomForest
from source.models.xg_boost import XGBoost
from source.models.neural_network import NeuralNetwork
from source.hyperparamters.grid_search import GridSearch
from source.pca import pca
import logging


def run(config):
    logger = logging.getLogger('pipeline.run')

    """DATA HANDLING"""
    logger.info('Data handling')
    if not config.getboolean('Features', 'use_prepared'):
        logger.info('Loading and preparing data')
        d = Data(config)
        d.download_data()
        d.load_data()

        f = Features(config, d.df_train, d.df_test)
    else:
        logger.info('Loading prepared data')
        f = Features(config)

    f.prepare_df_train()

    if config.getboolean('General', 'use_test'):
        f.prepare_df_test()

    X_train = f.df_train.drop(['hotel_cluster'], axis=1).to_numpy()
    pca(config, f.df_train.drop(['hotel_cluster'], axis=1))  # TESTING
    y_train = f.df_train['hotel_cluster'].to_numpy()
    if config.getboolean('General', 'use_test'):
        X_test = f.df_test.to_numpy()
        X_train, X_test = Features.scale_features(X_train, X_test)
    else:
        X_train = Features.scale_features(X_train)

    features = f.df_train.drop(['hotel_cluster'], axis=1).columns

    """TRAIN AND EVALUATE MODELS"""
    logger.info('Fit models')
    models = [
        {'model': DecisionTree, 'fitted': None},
        {'model': RandomForest, 'fitted': None},
        {'model': NeuralNetwork, 'fitted': None},
        {'model': XGBoost, 'fitted': None},
    ]

    for m in models:
        fitted_model = m['model'](config, X_train, y_train)
        fitted_model.train_model()
        fitted_model.calc_cross_val_score()
        m['fitted'] = fitted_model
        logger.info(f'{fitted_model.clf_name} score: {fitted_model.score}')

    """PERFORM GRID SEARCH"""
    if config.getboolean('GridSearch', 'perform_grid_search'):
        logger.info('Perform grid search')
        for m in models[:-2]:
            not_fitted_model = m['model']
            gs = GridSearch(config, not_fitted_model, X_train, y_train)
            gs.search()
            gs.print_best_results()
            del gs

    """DATA REMOVAL"""
    if config.get('Data', 'remove_after_run') == 'True':
        logger.info('Remove data from hard drive')
        d.remove_data()
