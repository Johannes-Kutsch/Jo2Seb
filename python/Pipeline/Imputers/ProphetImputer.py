import logging
import warnings
import pandas as pd
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin


class ProphetImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], max_gap_warning: None | int = 30):
        self.columns = columns
        self.max_gap_warning = max_gap_warning

        self.models_ = {}

    def fit(self, X, y=None):
        for col in self.columns:
            series = X[col].dropna()
            df = pd.DataFrame({"ds": series.index, "y": series.values})

            model = Prophet()
            logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
            model.fit(df)
            self.models_[col] = model

        return self

    def transform(self, X):
        X = X.copy()

        for col in self.columns:
            null_mask = X[col].isna()
            if not null_mask.any():
                continue

            self._warn_if_large_gap(col, null_mask)

            future = pd.DataFrame({"ds": X.index})
            forecast = self.models_[col].predict(future)
            forecast.index = X.index

            X.loc[null_mask, col] = forecast.loc[null_mask, "yhat"]

        return X

    def _warn_if_large_gap(self, col: str, null_mask: pd.Series):
        # Find consecutive NaN runs and check their length
        gap_lengths = (
            null_mask
            .astype(int)
            .groupby((null_mask != null_mask.shift()).cumsum())
            .sum()
        )
        max_gap = gap_lengths.max()
        if self.max_gap_warning and max_gap > self.max_gap_warning:
            warnings.warn(
                f"Column '{col}' has a gap of {max_gap} consecutive NaN values. "
                f"Prophet imputation may be unreliable for gaps larger than {self.max_gap_warning}.",
                UserWarning,
            )