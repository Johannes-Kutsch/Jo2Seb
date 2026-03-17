from sklearn.base import BaseEstimator, TransformerMixin

class DropHeadNaN(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.start_idx_ = None

    def fit(self, X, y=None):
        self.start_idx_ = X.notna().all(axis=1).idxmax()

        return self

    def transform(self, X):
        X = X.copy()
        return X.loc[self.start_idx_:]