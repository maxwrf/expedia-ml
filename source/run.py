import pandas as pd
from source.data.data import Data
from source.data.features import Features
from source.models.decision_tree import DecisionTree
from source.models.random_forest import RandomForest
from source.models.xg_boost import XGBoost
from source.hyperparamters.grid_search import GridSearch


def run(config):
    if not config.getboolean('Features', 'use_prepared'):
        d = Data(config)
        d.download_data()
        d.load_data()

        f = Features(config, d.df_train, d.df_test)
    else:
        f = Features(config)

    f.prepare_df_train()

    if config.getboolean('General', 'use_test'):
        f.prepare_df_test()

    X_train = f.df_train.drop(['hotel_cluster'], axis=1)
    y_train = f.df_train['hotel_cluster']
    # X_test = f.df_test

    models = [
        {'model': DecisionTree, 'fitted': None},
        {'model': RandomForest, 'fitted': None}
    ]

    for m in models:
        fitted_model = m['model'](config, X_train, y_train)
        fitted_model.train_model()
        fitted_model.calc_cross_val_score()
        m['fitted'] = fitted_model
        print(fitted_model.clf_name, fitted_model.score)

    if config.getboolean('GridSearch', 'perform_grid_search'):
        for m in models:
            not_fitted_model = m['model']
            gs = GridSearch(config, not_fitted_model, X_train, y_train)
            gs.search()
            gs.print_best_results()

    # TODO: Can the xg_boost ever take in all the features?
    forest = models[1]['fitted']
    feature_importances = pd.DataFrame(forest.clf.feature_importances_,
                                       index=X_train.columns,
                                       columns=['importance'])\
        .sort_values('importance', ascending=False)

    xgb = XGBoost(config,
                  X_train.loc[:, feature_importances.index[:50]],
                  y_train)
    xgb.train_model()
    xgb.calc_cross_val_score()
    print(xgb.score)

    if config.get('Data', 'remove_after_run') == 'True':
        d.remove_data()
