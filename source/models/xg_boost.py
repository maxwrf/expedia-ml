from xgboost import XGBClassifier
from source.models.base_model import BaseModel


class XGBoost(BaseModel):
    def __init__(self, config, X, y, params=None):
        super().__init__(config, X, y, params)
        self.clf_name = 'XGB Boost'

    def get_default_model(self):
        return XGBClassifier

    def get_default_parameter(self):
        """
        Defaults to objective='binary:logistic' and One vs rest
        """
        return {'n_jobs': -1,
                'max_depth': 2,
                'eta': 1,
                'random_state': 420}

    @staticmethod
    def get_grid_search_parameters():
        return [{'eta': [0.1, 1],
                 'n_jobs': [-1],
                 'objective': ['multi:softmax', 'binary:logistic']}]
