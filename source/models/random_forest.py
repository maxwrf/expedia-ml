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

    @staticmethod
    def get_grid_search_parameters():
        return [{'bootstrap': [True, False],
                 'max_depth': [10, 70],
                 'max_features': ['auto', 'sqrt'],
                 'min_samples_leaf': [1, 4],
                 'min_samples_split': [2, 10],
                 'n_estimators': [100, 2000]
                 }]
