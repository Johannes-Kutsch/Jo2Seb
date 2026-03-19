import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnFunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns:list[str], function=np.log1p, new_column_suffix:str=None):
        self.columns = columns
        self.function = function
        self.new_column_suffix = new_column_suffix

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        if self.new_column_suffix is None:
            X[self.columns] = X[self.columns].apply(self.function, axis=0)
            return X

        X[self.columns + [self.new_column_suffix]] = X[self.columns].apply(self.function, axis=1)
        return X

