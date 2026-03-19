from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

def create_xgboost():
    return Pipeline([
        ("model", XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])