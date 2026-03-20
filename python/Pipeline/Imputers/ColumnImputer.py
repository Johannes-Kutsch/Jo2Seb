from sklearn.base import BaseEstimator, TransformerMixin


class ColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], imputer):
        self.columns = columns
        self.imputer = imputer

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = self.imputer.transform(X[self.columns])
        return X