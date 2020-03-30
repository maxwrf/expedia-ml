from xgboost import XGBClassifier
from source.models.base_model import BaseModel


class XGBoost(BaseModel):
    def __init__(self, config, X, y, params=None):
        super().__init__(config, X, y, params)
        self.clf_name = 'XGB Boost'
        self.config = config

    def get_default_model(self):
        return XGBClassifier

    def get_default_parameter(self):
        """
        Defaults to objective='binary:logistic' and One vs rest
        """
        return {'n_jobs': -1,
                'max_depth': 3,
                'eta': .3,
                }

    def train_model(self):
        if self.clf is not None:
            self.clf.fit(self.X,
                         self.y,
                         eval_set=[(self.X, self.y)],
                         early_stopping_rounds=2)
        else:
            raise Exception('Model not defined.')

    @staticmethod
    def get_grid_search_parameters():
        return [{}]
