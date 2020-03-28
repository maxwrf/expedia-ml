from sklearn.tree import DecisionTreeClassifier
from source.models.base_model import BaseModel


class DecisionTree(BaseModel):
    def __init__(self, config, X, y, params=None):
        super().__init__(config, X, y, params)
        self.clf_name = 'Decision Tree'

    def get_default_model(self):
        return DecisionTreeClassifier

    def get_default_parameter(self):
        return {'max_depth': 10, 'random_state': 420}
