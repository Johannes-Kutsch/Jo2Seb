from sklearn.base import BaseEstimator, RegressorMixin, clone
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, DecomposeResult


class MSTLRegressor(BaseEstimator, RegressorMixin):
    """
        Regression model based on MSTL decomposition.

        The model decomposes the target time series into trend, seasonal,
        and residual components using Multiple Seasonal-Trend decomposition
        using LOESS (MSTL). Each component is then handled separately:

        - The trend component is modeled using a user-provided regression model.
        - Seasonal components are forecasted by repeating the last observed
          seasonal patterns.
        - Residuals are modeled using an optional external regression model
          that can incorporate exogenous features.

        The final prediction is obtained by summing the forecasts of the
        trend, seasonal, and residual components.

        Parameters
        ----------
        seasonal_periods : tuple of int, default=(24, 168)
            Seasonal periods used for MSTL decomposition. Each value represents
            the length of a seasonal cycle (e.g. 24 for daily seasonality in
            hourly data).

        trend_model : estimator
            Regression model used to forecast the trend component. Must
            implement `fit(X, y)` and `predict(X)`.

        resid_model : estimator, optional
            Regression model used to predict the residual component using
            exogenous features. Must implement `fit(X, y)` and `predict(X)`.
            If None, residuals are assumed to be zero during forecasting.

        Attributes
        ----------
        mstl_ : MSTL
            Fitted MSTL decomposition object.

        decomp_ : DecomposeResult
            Result of the MSTL decomposition containing trend, seasonal,
            and residual components.

        trend_model_ : estimator
            Fitted clone of the provided trend model.

        resid_model_ : estimator or None
            Fitted clone of the residual model if provided.

        seasonal_patterns_ : dict
            Dictionary containing the last observed seasonal pattern
            for each seasonal period.

        trend_index_ : int
            Index position of the last observed trend value used to
            generate future trend forecasts.
        """
    def __init__(self, seasonal_periods=None, trend_model=None, resid_model=None):
        self.seasonal_periods = seasonal_periods
        self.trend_model = trend_model
        self.resid_model = resid_model

        self.decomp_ = None

        self.trend_model_ = None
        self.resid_model_ = None

        self.seasonal_patterns_ = None
        self.trend_index_ = None


    def fit(self, X, y):
        """
        Fit the MSTLRegressor.

        The method performs the following steps:

        1. Decomposes the target series using MSTL.
        2. Fits the provided trend model on the extracted trend component.
        3. Stores the last seasonal patterns for each seasonal period.
        4. Optionally fits the residual regression model using the
           provided exogenous features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Exogenous features used to model the residual component.

        y : array-like of shape (n_samples,)
            Target time series.

        Returns
        -------
        self : MSTLRegressor
            Fitted estimator.
        """
        y = pd.Series(y)

        self.decomp_ = MSTL(y, periods=self.seasonal_periods).fit()

        self.seasonal_patterns_ = self._create_seasonal_patterns(self.decomp_.seasonal, self.seasonal_periods)

        self.trend_model_, self.trend_index_ = self._create_fitted_trend_model(self.decomp_.trend, self.trend_model)
        self.resid_model_ = self._create_fitted_resid_model(X, self.decomp_.resid, self.resid_model)

        return self

    def predict(self, X):
        """
        Generate forecasts for the given exogenous features.

        The prediction is computed as the sum of the forecasted trend,
        seasonal, and residual components.

        Parameters
        ----------
        X : array-like of shape (n_steps, n_features)
            Exogenous features for the forecast horizon.

        Returns
        -------
        ndarray of shape (n_steps,)
            Forecasted values of the target time series.
        """
        steps = len(X)

        t_future = np.arange(self.trend_index_, self.trend_index_ + steps).reshape(-1, 1)
        trend_forecast = self.trend_model_.predict(t_future)

        seasonal_forecast = self._forecast_seasonality(steps, self.seasonal_patterns_)

        if self.resid_model_ is not None:
            resid_forecast = self.resid_model_.predict(X)
        else:
            resid_forecast = np.zeros(steps)

        return trend_forecast + seasonal_forecast + resid_forecast

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
        for i, p in enumerate(seasonal_periods):

            if hasattr(seasonal, "iloc"):
                col = seasonal.columns[i]
                pattern = seasonal[col].iloc[-p:].values
            else:
                pattern = seasonal.iloc[-p:].values

            seasonal_patterns[p] = pattern

        return seasonal_patterns

    @staticmethod
    def _forecast_seasonality(steps, seasonal_patterns):
        seasonal_forecast = np.zeros(steps)

        for p, pattern in seasonal_patterns.items():
            repeated = np.tile(
                pattern,
                int(np.ceil(steps / p))
            )[:steps]

            seasonal_forecast += repeated

        return seasonal_forecast