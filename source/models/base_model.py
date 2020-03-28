from abc import abstractmethod
import numpy as np
from sklearn.model_selection import cross_val_score


class BaseModel():
    def __init__(self, config, X, y, params):
        self.config = config
        self.X = X
        self.y = y
        self.cv_folds = config.getint('Models', 'cvFolds')
        self.scoring = config.get('Models', 'scoring')
        self.clf = None
        self.clf_name = 'base_model'
        self.best_grid_search_results = None
        self.best_grid_search_model = None
        self.intizalize_model(params)

    def intizalize_model(self, params):
        if not params:
            params = self.get_default_parameter()
        model = self.get_default_model()
        self.clf = model(**params)

    @abstractmethod
    def get_default_model(self):
        pass

    @abstractmethod
    def get_default_parameter(self):
        pass

    @abstractmethod
    def get_grid_search_parameters(self):
        pass

    def train_model(self):
        if self.clf is not None:
            self.clf.fit(self.X, self.y)
        else:
            raise Exception('Model not defined.')

    def calc_cross_val_score(self):
        score = np.mean(cross_val_score(self.clf,
                                        self.X,
                                        self.y,
                                        cv=self.cv_folds,
                                        scoring=self.scoring, n_jobs=-1))
        self.score = score
        return score

    def save_model(self):
        pass
