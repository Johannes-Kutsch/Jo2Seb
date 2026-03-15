import mlflow
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DAILY_EXPERIMENT_NAME = "daily"
HOURLY_EXPERIMENT_NAME = "hourly"
LOCAL_TRACKING_URI = "http://localhost:5000"

def start_local_experiment(experiment: str):
    if mlflow.get_tracking_uri() != LOCAL_TRACKING_URI:
        mlflow.set_tracking_uri(LOCAL_TRACKING_URI)

    mlflow.set_experiment(experiment)

def log_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric(f"mae", mae)
    mlflow.log_metric(f"rmse", rmse)
    mlflow.log_metric(f"r2", r2)