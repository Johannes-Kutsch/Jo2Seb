from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import logging
from prophet import Prophet

from Pipeline.Decomposer.ResidualOption import ResidualOption


class ProphetDecomposer(BaseEstimator, RegressorMixin):
    def __init__(self, column_name, residual_options: ResidualOption = ResidualOption.IGNORE, **prophet_kwargs):
        self.column_name = column_name
        self.residual_options = residual_options
        self.prophet_kwargs = prophet_kwargs

        self.model_ = None
        self.seasonal_columns_ = None
        self.seasonal_trend_columns = None

    def fit(self, X, y=None):
        df = self._to_prophet_df(X)

        self.model_ = Prophet(**self.prophet_kwargs)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        self.model_.fit(df)

        # Determine which seasonality columns Prophet produced
        forecast = self.model_.predict(df)
        self.seasonal_columns_ = self._get_seasonal_columns(forecast)

        # Track all component column names that will be written to X in transform
        self.seasonal_trend_columns = (
            [f"{self.column_name}_trend"] +
            [f"{self.column_name}_{col}" for col in self.seasonal_columns_]
        )

        return self

    def get_seasonal_trend_columns(self) -> list[str]:
        return self.seasonal_trend_columns

    def transform(self, X):
        X = X.copy()

        future = pd.DataFrame({"ds": X.index})
        forecast = self.model_.predict(future)
        forecast.index = X.index

        X[f"{self.column_name}_trend"] = forecast["trend"].values

        for col in self.seasonal_columns_:
            X[f"{self.column_name}_{col}"] = forecast[col].values

        seasonal_sum = forecast[self.seasonal_columns_].values.sum(axis=1)
        residuals = X[self.column_name] - forecast["trend"].values - seasonal_sum

        if self.residual_options == ResidualOption.OVERWRITE_ORIGINAL_FEATURE:
            X[self.column_name] = residuals
        elif self.residual_options == ResidualOption.NEW_FEATURE:
            X[f"{self.column_name}_residuals"] = residuals

        return X

    def _to_prophet_df(self, X):
        return pd.DataFrame({
            "ds": X.index,
            "y": X[self.column_name].values,
        })

    @staticmethod
    def _get_seasonal_columns(forecast: pd.DataFrame) -> list[str]:
        # Prophet adds additive/multiplicative term columns and individual seasonality
        # columns ending in a known suffix pattern. We pick all seasonality components
        # by excluding known non-seasonality columns.
        exclude = {
            "ds", "trend", "trend_lower", "trend_upper",
            "yhat", "yhat_lower", "yhat_upper",
            "additive_terms", "additive_terms_lower", "additive_terms_upper",
            "multiplicative_terms", "multiplicative_terms_lower", "multiplicative_terms_upper",
        }
        return [
            col for col in forecast.columns
            if col not in exclude
            and not col.endswith("_lower")
            and not col.endswith("_upper")
        ]