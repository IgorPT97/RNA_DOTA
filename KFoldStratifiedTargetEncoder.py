from sklearn import base, model_selection
import numpy as np

class KFoldStratifiedTargetEncoder(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, number_of_folds=5, verbose=False):
        self.number_of_folds = number_of_folds
        self.verbose = verbose

        self._values = None

    @property
    def number_of_folds(self):
        return self.__number_of_folds

    @number_of_folds.setter
    def number_of_folds(self, new_number_of_folds):
        self.__number_of_folds = new_number_of_folds

    @property
    def verbose(self):
        return self.__verbose

    @verbose.setter
    def verbose(self, new_verbose):
        self.__verbose = new_verbose

    def fit(self, X, y, **fit_params):
        encoded_X, current_fold = np.empty_like(X, dtype=np.float_), 0
        stratified_k_fold = model_selection.StratifiedKFold(n_splits=self.number_of_folds)

        for train_indices, test_indices in stratified_k_fold.split(X, y):
            X_train, y_train, X_test = X[train_indices, :], y[train_indices], X[test_indices, :]
            self._values = {column: None for column in range(X.shape[1])}

            for column in range(X.shape[1]):
                X_train_column, X_test_column = X_train[:, column], X_test[:, column]
                encoded_X_column = np.full_like(X_test_column, y_train.mean(), dtype=np.float_)
                categories = np.unique(X_train_column)

                for category in categories:
                    encoded_X_column[X_test_column == category] = y_train[X_train_column == category].mean()
                encoded_X[test_indices, column] = encoded_X_column

            current_fold += 1

        for column in range(X.shape[1]):
            X_column = X[:, column]
            categories = np.unique(X_column)

            self._values[column] = {category: None for category in categories}

            for category in categories:
                self._values[column][category] = y[X_column == category].mean()

        if "return_encoded_X" in fit_params.keys() and fit_params["return_encoded_X"] is True:
            return encoded_X

        return self

    def transform(self, X):
        if self._values is None:
            raise ValueError()

        encoded_X = np.empty_like(X, dtype=np.float_)

        for column in range(X.shape[1]):
            X_column = X[:, column]
            encoded_X_column = np.empty_like(X_column, dtype=np.float_)

            for category, value in self._values[column].items():
                encoded_X_column[X_column == category] = value

            encoded_X[:, column] = encoded_X_column

        return encoded_X

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError()
        else:
            return self.fit(X, y, return_encoded_X=True)
