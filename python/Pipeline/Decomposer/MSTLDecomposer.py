from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, DecomposeResult

from enum import Enum

class ResidualOptions(Enum):
    IGNORE = "ignore"
    OVERWRITE_ORIGINAL_FEATURE = "overwrite_original_feature"
    NEW_FEATURE = "new_feature"

class MSTLDecomposer(BaseEstimator, RegressorMixin):
    def __init__(self, column_name, seasonal_periods=None, trend_model=None, residual_options:ResidualOptions = ResidualOptions.IGNORE):
        self.seasonal_periods = seasonal_periods
        self.trend_model = trend_model
        self.column_name = column_name
        self.residual_options = residual_options

        self.decomp_ = None

        self.trend_model_ = None

        self.seasonal_patterns_ = None
        self.trend_index_ = None


    def fit(self, X, y=None):
        series = X[self.column_name]

        self.decomp_ = MSTL(series, periods=self.seasonal_periods).fit()

        self.seasonal_patterns_ = self._create_seasonal_patterns(self.decomp_.seasonal, self.seasonal_periods)
        self.trend_model_, self.trend_index_ = self._create_fitted_trend_model(self.decomp_.trend, self.trend_model)

        return self

    def transform(self, X):
        X = X.copy()
        steps = len(X)

        trend_forecast_index = np.arange(self.trend_index_, self.trend_index_ + steps).reshape(-1, 1)
        trend_feature = self.trend_model_.predict(trend_forecast_index)
        X[f"{self.column_name}_trend"] = trend_feature

        seasonal_features = self._forecast_seasonality(steps, self.seasonal_patterns_)
        for col in seasonal_features.columns:
            X[f"{self.column_name}_{col}"] = seasonal_features[col].values

        residuals = X[self.column_name] - trend_feature - seasonal_features.values.sum(axis=1)

        if self.residual_options == ResidualOptions.OVERWRITE_ORIGINAL_FEATURE:
            X[self.column_name] = residuals
        elif self.residual_options == ResidualOptions.NEW_FEATURE:
            X[f"{self.column_name}_residuals"] = residuals

        return X

    @staticmethod
    def _create_fitted_resid_model(X, y_resid, orig_resid_model: RegressorMixin):
        if orig_resid_model is None:
            return None

        resid_model = clone(orig_resid_model)
        resid_model.fit(X, y_resid)

        return resid_model

    @staticmethod
    def _create_fitted_trend_model(y_trend, orig_trend_model: RegressorMixin):
        if orig_trend_model is None:
            raise ValueError("trend_model must be provided")

        trend_model = clone(orig_trend_model)

        t = np.arange(len(y_trend)).reshape(-1, 1)
        trend_model.fit(t, y_trend)

        trend_index = len(y_trend)

        return trend_model, trend_index

    @staticmethod
    def _create_seasonal_patterns(seasonal, seasonal_periods):
        seasonal_patterns = {}
        if isinstance(seasonal, pd.DataFrame):
            for i, p in enumerate(seasonal_periods):
                col = seasonal.columns[i]
                pattern = seasonal[col].iloc[-p:].values
                seasonal_patterns[p] = pattern
        elif isinstance(seasonal, pd.Series):
            p = seasonal_periods[0]
            pattern = seasonal.iloc[-p:].values
            seasonal_patterns[p] = pattern
        else:
            raise TypeError("Unexpected type for seasonal")

        return seasonal_patterns

    @staticmethod
    def _forecast_seasonality(steps, seasonal_patterns):
        seasonal_features = {}

        for p, pattern in seasonal_patterns.items():
            seasonal_features[f"seasonal_{p}"] = (
                np.tile(pattern, int(np.ceil(steps / p)))[:steps]
            )

        return pd.DataFrame(seasonal_features)