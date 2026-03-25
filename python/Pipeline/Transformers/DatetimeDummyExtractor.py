from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeDummyExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts comprehensive date/time features from a DataFrame's DatetimeIndex.
    'month' and 'weekday' can optionally be returned as categorical strings.

    Parameters
    ----------
    features : list of str, optional
        List of features to extract. Possible values:
        'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
        'quarter', 'dayofyear', 'is_weekend'.
        Default is all features.
    month_as_category : bool, default=True
        If True, month numbers are converted to string names (e.g., 'Jan', 'Feb').
    weekday_as_category : bool, default=True
        If True, weekdays are converted to string names (e.g., 'Mon', 'Tue').
    """

    ALL_FEATURES = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
                    'quarter', 'dayofyear', 'is_weekend']

    MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, features=None, month_as_category=True, weekday_as_category=True):
        self.features = features or self.ALL_FEATURES
        self.month_as_category = month_as_category
        self.weekday_as_category = weekday_as_category

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = X.index

        if not hasattr(dt, 'month'):
            raise TypeError(f"Index must be a DatetimeIndex, got {type(dt).__name__}.")

        if 'year' in self.features:
            X['year'] = dt.year
        if 'month' in self.features:
            if self.month_as_category:
                X['month'] = [self.MONTH_NAMES[m - 1] for m in dt.month]
            else:
                X['month'] = dt.month
        if 'day' in self.features:
            X['day'] = dt.day
        if 'weekday' in self.features:
            if self.weekday_as_category:
                X['weekday'] = [self.WEEKDAY_NAMES[w] for w in dt.weekday]
            else:
                X['weekday'] = dt.weekday
        if 'hour' in self.features:
            X['hour'] = dt.hour
        if 'minute' in self.features:
            X['minute'] = dt.minute
        if 'second' in self.features:
            X['second'] = dt.second
        if 'quarter' in self.features:
            X['quarter'] = dt.quarter
        if 'dayofyear' in self.features:
            X['dayofyear'] = dt.dayofyear
        if 'is_weekend' in self.features:
            X['is_weekend'] = dt.weekday >= 5

        return X