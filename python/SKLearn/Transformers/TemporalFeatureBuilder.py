from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, lags=None, rolling_windows=None, roll_shift = 1, drop_original=False):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.column_name = column_name
        self.drop_original = drop_original
        self.roll_shift = roll_shift

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if self.lags:
            for lag in self.lags:
                X[f"{self.column_name}_lag_{lag}"] = X[self.column_name].shift(lag)

        if self.rolling_windows:
            for window in self.rolling_windows:
                X[f"{self.column_name}_roll_mean_{window}"] = X[self.column_name].shift(self.roll_shift).rolling(window).mean()
                X[f"{self.column_name}_roll_std_{window}"] = X[self.column_name].shift(self.roll_shift).rolling(window).std()

        if self.drop_original:
            X.drop(columns=[self.column_name], inplace=True)

        return X