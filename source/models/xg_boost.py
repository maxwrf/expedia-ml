from xgboost import XGBClassifier
from source.models.base_model import BaseModel


class XGBoost(BaseModel):
    def __init__(self, config, X, y, params=None):
        super().__init__(config, X, y, params)
        self.clf_name = 'XGB Boost'

    def get_default_model(self):
        return XGBClassifier

    def get_default_parameter(self):
        return {'n_jobs': -1, 'max_depth': 2, 'eta': 1, 'random_state': 420}
