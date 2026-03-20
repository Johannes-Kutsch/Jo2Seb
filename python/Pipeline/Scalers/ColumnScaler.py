from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class ColumnScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], scaler=None):
        self.columns = columns
        self.scaler = scaler if scaler is not None else RobustScaler()

        self.scaler_ = None

    def fit(self, X, y=None):
        from sklearn.base import clone
        self.scaler_ = clone(self.scaler)
        self.scaler_.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.scaler_.transform(X[self.columns])
        return X