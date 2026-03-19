from sklearn.base import BaseEstimator, RegressorMixin, clone


class AlignedModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model: RegressorMixin):
        self.model = model
        self.model_ = None

    def fit(self, X, y=None):
        y_aligned = y.loc[X.index] if y is not None else None

        self.model_ = clone(self.model)
        self.model_.fit(X, y_aligned)

        return self

    def predict(self, X):
        return self.model_.predict(X)