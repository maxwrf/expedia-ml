import pandas as pd
from source.data.data import Data
from source.data.features import Features
from source.models.decision_tree import DecisionTree
from source.models.random_forest import RandomForest
from source.models.xg_boost import XGBoost


def run(config):
    if not config.getboolean('Features', 'use_prepared'):
        d = Data(config)
        d.download_data()
        d.load_data()

        f = Features(config, d.df_train, d.df_test)
    else:
        f = Features(config)
    f.prepare_df_test()
    f.prepare_df_train()

    X_train = f.df_train.drop(['hotel_cluster'], axis=1)
    y_train = f.df_train['hotel_cluster']
    # X_test = f.df_test

    tree = DecisionTree(config, X_train, y_train)
    tree.train_model()
    tree.calc_cross_val_score()
    print(tree.score)

    forest = RandomForest(config, X_train, y_train)
    forest.train_model()
    forest.calc_cross_val_score()
    print(forest.score)

    # TODO: Can the xg_boost ever take in all the features?
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
