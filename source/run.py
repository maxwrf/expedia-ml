import pandas as pd
from source.data.data import Data
from source.data.features import Features
from source.models.decision_tree import DecisionTree
from source.models.random_forest import RandomForest
from source.models.xg_boost import XGBoost
from source.models.neural_network import NeuralNetwork
from source.hyperparamters.grid_search import GridSearch
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
    y_train = f.df_train['hotel_cluster'].to_numpy()
    if config.getboolean('General', 'use_test'):
        X_test = f.df_test.to_numpy()
        X_train, X_test = Features.scale_features(X_train, X_test)
    else:
        X_train = Features.scale_features(X_train)

    features = f.df_train.drop(['hotel_cluster'], axis=1).columns

    """MODELS AND GRID SEARCH"""
    logger.info('Fit models')
    models = [
        {'model': DecisionTree, 'fitted': None},
        {'model': RandomForest, 'fitted': None},
        {'model': NeuralNetwork, 'fitted': None},
    ]

    for m in models:
        fitted_model = m['model'](config, X_train, y_train)
        fitted_model.train_model()
        fitted_model.calc_cross_val_score()
        m['fitted'] = fitted_model
        logger.info(f'{fitted_model.clf_name} | Score: {fitted_model.score}')

    logger.info('Perform grid search')
    if config.getboolean('GridSearch', 'perform_grid_search'):
        for m in models[:-1]:
            not_fitted_model = m['model']
            gs = GridSearch(config, not_fitted_model, X_train, y_train)
            gs.search()
            gs.print_best_results()
            del gs

    # TODO: Can the xg_boost ever take in all the features?
    tree = models[0]['fitted']
    feature_importances = pd.DataFrame(tree.clf.feature_importances_,
                                       index=features,
                                       columns=['importance'])\
        .sort_values('importance', ascending=False)
    X_train_xgb = f.df_train.drop(['hotel_cluster'], axis=1)\
        .loc[:, feature_importances.index[:50]].to_numpy()
    xgb = XGBoost(config,
                  X_train_xgb,
                  y_train)
    xgb.train_model()
    xgb.calc_cross_val_score()
    logger.info(f'XGB | Score: {xgb.score}')

    if config.getboolean('GridSearch', 'perform_grid_search'):
        gs = GridSearch(config,
                        XGBoost,
                        X_train_xgb,
                        y_train)
        gs.search()
        gs.print_best_results()

    """DATA REMOVAL"""
    logger.info('Remove data from hard drive')
    if config.get('Data', 'remove_after_run') == 'True':
        d.remove_data()
