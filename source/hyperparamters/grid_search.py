import gc
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import logging


class GridSearch():
    logger = logging.getLogger('pipeline.run.grid_search')

    def __init__(self, config, model, X, y):
        self.config = config
        self.model = model
        self.X = X
        self.y = y
        self.cv_folds = config.getint('Models', 'cvFolds')
        self.results = []

    def search(self):
        params = self.model.get_grid_search_parameters()
        grid = list(ParameterGrid(params))

        # TODO: How can we parllelize this?
        scores = Parallel(n_jobs=-1)(delayed(self.calc)(param_comb) for param_comb in grid)

        for score, param_comb in zip(scores, grid):
            GridSearch.logger.info(f'{param_comb} | {score}')
            self.results.append({'params': param_comb, 'score': score})

    def calc(self, param_comb):
        clf = self.model(self.config, self.X, self.y, param_comb)
        score = clf.calc_cross_val_score()
        del clf.X
        del clf.y
        del clf
        gc.collect()
        GridSearch.logger.info(f'{param_comb} | {score}')
        return score

    def get_best_result(self):
        return max(self.results,
                   key=lambda res_dict: res_dict['score']).values()

    def print_best_results(self):
        params, score = self.get_best_result()
        GridSearch.logger.info(f'Best model after Grid Search | Params: {params},\
                score: {score}')

    def get_best_model(self):
        params, _ = self.get_best_result()
        clf = self.clf(self.config, self.X, self.y, params)
        clf.train_model()
        return clf
