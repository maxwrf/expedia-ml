from source.data.data import Data
from source.data.features import Features
from source.models.decision_tree import DecisionTree
from source.models.random_forest import RandomForest
from source.models.xg_boost import XGBoost
from source.models.neural_network import NeuralNetwork
from source.hyperparamters.grid_search import GridSearch
from source.pca import pca
from source.data_leakage import Exploit
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging
from IPython import embed


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
    f.df_train.drop('Unnamed: 0', axis=1, inplace=True)

    if config.getboolean('General', 'use_test'):
        f.prepare_df_test()

    X_train = f.df_train.drop(['hotel_cluster'], axis=1)
    pca(config, f.df_train.drop(['hotel_cluster'], axis=1))
    y_train = f.df_train['hotel_cluster']

    if config.getboolean('General', 'use_test'):
        X_test = f.df_test.to_numpy()
        X_train, X_test = Features.scale_features(X_train, X_test)
    else:
        logger.info('train, holdout, test split is 98% | 1% | 1%')
        X_train, X_holdout_test, y_train, y_holdout_test = train_test_split(
            X_train, y_train, test_size=0.02, random_state=420)
        X_holdout, X_test, y_holdout, y_test = train_test_split(
            X_holdout_test, y_holdout_test, test_size=0.55, random_state=420)
        X_columns = X_train.columns
        X_train, X_holdout, X_test = Features.scale_features(X_train,
                                                             X_holdout, X_test)

        exploit = Exploit(X_train, y_train, X_columns)

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
        m['fitted'] = fitted_model

        # check config whether to perfrom cross fold validation
        # or simply use the hold out set
        if config.getboolean('General', 'cross_validation'):
            fitted_model.calc_cross_val_score()
        else:
            preds_holdout = fitted_model.predict(X_holdout, exploit)
            fitted_model.score = accuracy_score(y_holdout, preds_holdout)

        logger.info(f'{fitted_model.clf_name} score: {fitted_model.score}')

    # Calculate final score for the best model
    best_model = max(models, key=lambda model: model['fitted'].score)
    preds_test = best_model['fitted'].predict(X_test)
    best_model_test_score = accuracy_score(y_test, preds_test)
    logger.info(f"Best model: {best_model['fitted'].clf_name} \
                  test score: {best_model_test_score}")

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
