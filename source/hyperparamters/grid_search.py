from sklearn.model_selection import ParameterGrid


class GridSearch():
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

        for param_comb in grid:
            score = self.model(self.config, self.X, self.y, param_comb)\
                .calc_cross_val_score()
            self.results.append({'params': param_comb, 'score': score})

    def get_best_result(self):
        return max(self.results,
                   key=lambda res_dict: res_dict['score']).values()

    def print_best_results(self):
        params, score = self.get_best_result()
        print(f'Best model after Grid Search | Params: {params},\
                score: {score}')

    def get_best_model(self):
        params, _ = self.get_best_result()
        clf = self.clf(self.config, self.X, self.y, params)
        clf.train_model()
        return clf
