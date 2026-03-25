from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeDummyExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts comprehensive date/time features from a DataFrame's DatetimeIndex.
    'month' and 'weekday' can optionally be returned as one-hot dummy columns.

    Parameters
    ----------
    features : list of str, optional
        List of features to extract. Possible values:
        'year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
        'quarter', 'dayofyear', 'is_weekend'.
        Default is all features.
    month_as_dummies : bool, default=True
        If True, creates binary columns is_jan, is_feb, ..., is_dec.
        If False, adds a single numeric 'month' column (1–12).
    weekday_as_dummies : bool, default=True
        If True, creates binary columns is_monday, is_tuesday, ..., is_sunday.
        If False, adds a single numeric 'weekday' column (0=Mon … 6=Sun).
    drop_first : bool, default=False
        If True, drops the first dummy column per group to avoid multicollinearity
        (e.g. is_jan and is_monday are dropped).
    """

    ALL_FEATURES = ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'second',
                    'quarter', 'dayofyear', 'is_weekend']

    MONTH_NAMES  = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    WEEKDAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday',
                     'friday', 'saturday', 'sunday']

    def __init__(self, features=None, month_as_dummies=True,
                 weekday_as_dummies=True, drop_first=False):
        self.features          = features or self.ALL_FEATURES
        self.month_as_dummies  = month_as_dummies
        self.weekday_as_dummies = weekday_as_dummies
        self.drop_first        = drop_first

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = X.index

        if not hasattr(dt, 'month'):
            raise TypeError(f"Index must be a DatetimeIndex, got {type(dt).__name__}.")

        if 'year' in self.features:
            X['year'] = dt.year.astype(float)

        if 'month' in self.features:
            if self.month_as_dummies:
                names = self.MONTH_NAMES[1:] if self.drop_first else self.MONTH_NAMES
                for i, name in enumerate(names):
                    month_num = i + 2 if self.drop_first else i + 1
                    X[f'is_{name}'] = (dt.month == month_num).astype(float)
            else:
                X['month'] = dt.month.astype(float)

        if 'day' in self.features:
            X['day'] = dt.day.astype(float)

        if 'weekday' in self.features:
            if self.weekday_as_dummies:
                names = self.WEEKDAY_NAMES[1:] if self.drop_first else self.WEEKDAY_NAMES
                for i, name in enumerate(names):
                    wd_num = i + 1 if self.drop_first else i
                    X[f'is_{name}'] = (dt.weekday == wd_num).astype(float)
            else:
                X['weekday'] = dt.weekday.astype(float)

        if 'hour' in self.features:
            X['hour'] = dt.hour.astype(float)
        if 'minute' in self.features:
            X['minute'] = dt.minute.astype(float)
        if 'second' in self.features:
            X['second'] = dt.second.astype(float)
        if 'quarter' in self.features:
            X['quarter'] = dt.quarter.astype(float)
        if 'dayofyear' in self.features:
            X['dayofyear'] = dt.dayofyear.astype(float)
        if 'is_weekend' in self.features:
            X['is_weekend'] = (dt.weekday >= 5).astype(float)

        return X