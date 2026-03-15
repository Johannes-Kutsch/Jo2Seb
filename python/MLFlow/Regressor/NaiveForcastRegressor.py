import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveForcastRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self._last_value = None

    def fit(self, X, y):
        self._last_value = y.iloc[-1]
        return self

    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self._last_value)

    def __sklearn_is_fitted__(self):
        return self._last_value is not None