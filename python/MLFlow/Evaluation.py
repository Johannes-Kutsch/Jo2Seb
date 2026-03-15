import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from typing import Optional, Any

from MLFlow.Utils import log_metrics, start_local_experiment


def evaluate_timeseries_pipeline(X:DataFrame, y:Series, pipeline:Pipeline, experiment:str, train_test_split:float = 0.8, model_name:str = "", params:Optional[dict[str, Any]] = None, walk_forward:bool = False):
    """
    Evaluates a scikit-learn pipeline on a time series using TimeSeriesSplit.

    Parameters:
    - X: pd.DataFrame, Feature matrix
    - y: pd.Series, Target variable
    - pipeline: sklearn Pipeline or estimator with fit/predict
    - n_splits: int, number of splits for TimeSeriesSplit
    - model_name: str, name of the model

    Returns:
    - results: list of dicts with split info and metrics
    """

    X_test, X_train, y_test, y_train = create_train_test_split(X, y, train_test_split)

    if params is None:
        params = {}

    params["train_test_split"] = str(train_test_split)
    params["features"] = X.columns.tolist()
    params["walk_forward"] = str(walk_forward)

    params["pipeline_structure"] = " -> ".join([type(transformer).__name__ for _, transformer in pipeline.steps])
    for name, transformer in pipeline.steps: ## Save Hyperparameters
        transformer_params = transformer.get_params()
        for k, v in transformer_params.items():
            if v is not None:
                params[f"{name}__{k}"] = str(v)

    start_local_experiment(experiment)

    with mlflow.start_run(run_name=f"{model_name}"):
        if walk_forward:
            y_pred = walk_forward_predict(X_train, y_train, X_test, y_test, pipeline)
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

        pred_nan_mask = ~np.isnan(y_pred)
        y_pred = y_pred[pred_nan_mask]
        y_test = y_test[pred_nan_mask]

        log_metrics(y_test, y_pred)
        signature = infer_signature(X_test[pred_nan_mask], y_pred)
        mlflow.sklearn.log_model(sk_model=pipeline, name=model_name, signature=signature, params=params)

    return pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

def walk_forward_predict(X_train, y_train, X_test, y_test, pipeline):

    y_pred = pd.Series(np.nan, index=y_test.index)

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    train_size = len(X_train)

    for i in range(len(X_test)):
        y_pred.iloc[i] = predict_at_position(X_all, y_all, pipeline, train_size + i)

    return y_pred


def predict_at_position(X_all: DataFrame, y_all: DataFrame, pipeline, position: int) -> float:
    X_train_step = X_all.iloc[:position]
    y_train_step = y_all.iloc[:position]

    pipeline.fit(X_train_step, y_train_step)

    return pipeline.predict(X_all.iloc[[position]])[0]


def create_train_test_split(X: DataFrame, y: Series, train_test_split: float):
    split = int(len(y) * train_test_split)
    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]
    return X_test, X_train, y_test, y_train


def plot_prediction(df, model_name:str, x_label:str = 'Date', y_label:str = 'PM10 [µg/m³]'):
    plt.figure(figsize=(24,10))
    plt.plot(df.index, df['y_test'], label='original', color='black', lw=2)
    plt.plot(df.index, df['y_pred'], label='predicted', color='red')
    plt.xlabel(f"\n{x_label}", fontsize=20, fontweight="bold")
    plt.ylabel(f"{y_label}\n", fontsize=20, fontweight="bold")
    plt.title(f"\n{model_name} {y_label} Prediction vs Original\n", fontsize=24, fontweight="bold")
    plt.legend(fontsize=20)
    plt.show()