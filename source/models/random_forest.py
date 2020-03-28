from sklearn.ensemble import RandomForestClassifier
from source.models.base_model import BaseModel


class RandomForest(BaseModel):
    def __init__(self, config, X, y, params=None):
        super().__init__(config, X, y, params)
        self.clf_name = 'Random Forest'

    def get_default_model(self):
        return RandomForestClassifier

    def get_default_parameter(self):
        return {'random_state': 420}
