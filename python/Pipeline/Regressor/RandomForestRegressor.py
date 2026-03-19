from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def create_random_forest():
    return Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])