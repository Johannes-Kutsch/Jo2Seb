from sklearn.base import BaseEstimator, RegressorMixin, clone


class SeasonalTrendRecomposer(BaseEstimator, RegressorMixin):
    def __init__(self, model: RegressorMixin, decomposer = None, component_columns: list[str] = None):
        self.model = model
        self.decomposer = decomposer
        self.component_columns = component_columns

        self.model_ = None
        self.component_columns_ = None

    def fit(self, X, y=None):
        if self.decomposer is not None:
            self.component_columns_ = self.decomposer.get_component_columns()
        elif self.component_columns is not None:
            self.component_columns_ = self.component_columns
        else:
            raise ValueError("Either decomposer or component_columns must be provided")

        y_aligned = y.loc[X.index] if y is not None else None

        self.model_ = clone(self.model)
        self.model_.fit(X, y_aligned)

        return self

    def predict(self, X):
        residual_predictions = self.model_.predict(X)
        component_sum = X[self.component_columns_].values.sum(axis=1)
        return residual_predictions + component_sum