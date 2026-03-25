import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from typing import Optional, Any
import matplotlib.dates as mdates

from Evaluation.Utils import log_metrics, start_local_experiment


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

    if params is None:
        params = {}

    params["train_test_split"] = str(train_test_split)
    params["features"] = X.columns.tolist()
    params["walk_forward"] = str(walk_forward)

    params["pipeline_structure"] = " -> ".join([type(transformer).__name__ for _, transformer in pipeline.steps])
    for name, transformer in pipeline.steps:
        transformer_params = transformer.get_params()
        for k, v in transformer_params.items():
            if v is not None:
                params[f"{name}__{k}"] = str(v)

    start_local_experiment(experiment)

    with mlflow.start_run(run_name=f"{model_name}"):
        X_test_transformed, y_pred, y_test = (
            _walk_forward_predict(X, y, pipeline, train_test_split)
            if walk_forward
            else _predict(X, y, pipeline, train_test_split)
        )
        log_metrics(y_test, y_pred)
        signature = infer_signature(X_test_transformed, y_pred)
        mlflow.sklearn.log_model(sk_model=pipeline, name=model_name, signature=signature, params=params)

    return pd.DataFrame({"y_test": y_test, "y_pred": y_pred})


def _transform_all(pipeline: Pipeline, X: DataFrame) -> DataFrame:
    X_transformed = X
    for _, transformer in pipeline.steps:
        if isinstance(transformer, Pipeline):
            X_transformed = _transform_all(transformer, X_transformed)
        else:
            X_transformed = transformer.transform(X_transformed)
    return X_transformed


def _transform_without_final(pipeline: Pipeline, X: DataFrame) -> DataFrame:
    X_transformed = X
    for _, transformer in pipeline.steps[:-1]:
        if isinstance(transformer, Pipeline):
            X_transformed = _transform_all(transformer, X_transformed)
        else:
            X_transformed = transformer.transform(X_transformed)
    return X_transformed


def _predict(X: DataFrame, y: Series, pipeline: Pipeline, train_test_split: float):
    split = int(len(X) * train_test_split)
    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    pipeline.fit(X_train, y_train)

    X_transformed = _transform_without_final(pipeline, X)
    X_test_transformed = X_transformed.loc[X.index[split]:]

    y_test = y.loc[X_test_transformed.index]

    y_pred = pipeline[-1].predict(X_test_transformed)
    return X_test_transformed, y_pred, y_test


def _walk_forward_predict(X: DataFrame, y: Series, pipeline: Pipeline, train_test_split: float):
    split = int(len(X) * train_test_split)

    y_pred_list = []
    y_test_list = []
    X_test_transformed_last = None

    progress_bar = tqdm(total=len(X) - split, desc="Walk-forward prediction", ncols=120)

    for i in range(len(X) - split):
        position = split + i
        X_train_step = X.iloc[:position]
        y_train_step = y.iloc[:position]

        pipeline.fit(X_train_step, y_train_step)

        X_transformed = _transform_without_final(pipeline, X.iloc[:position + 1])
        X_test_transformed_last = X_transformed.iloc[[-1]]

        y_pred_list.append(pipeline[-1].predict(X_test_transformed_last)[0])
        y_test_list.append(y.iloc[position])

        progress_bar.update(1)

    progress_bar.close()

    index = X.index[split:]
    return (
        X_test_transformed_last,
        pd.Series(y_pred_list, index=index),
        pd.Series(y_test_list, index=index),
    )

def plot_prediction(df, model_name: str, x_label: str | None = 'Date', y_label: str | None = 'PM10 [µg/m³]', bg_color ='#262626', text_color ='#ffffff', original_color ="#d4d4d4", pred_color ='#ff4d4d', labelpad = 15,
                    title_font_size = 30, font_size = 20, legend_font_size = 20, figsize = (24, 10), ax_lw = 2, lw = 2, x_tick_interval = 3, legend = True, pred_label = "predicted", orig_label = "original"):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.plot(df.index, df['y_test'], label=orig_label, color=original_color, lw = lw)
    ax.plot(df.index, df['y_pred'], label=pred_label, color=pred_color, lw = lw)

    if x_label is not None:
        ax.set_xlabel(f"{x_label}", fontsize=font_size, fontweight="bold", color=text_color, labelpad=labelpad)
    if y_label is not None:
        ax.set_ylabel(f"{y_label}", fontsize=font_size, fontweight="bold", color=text_color, labelpad=labelpad)
    #ax.set_title(f"{model_name}\n", fontsize=title_font_size, fontweight="bold", color=text_color)

    ax.tick_params(axis='both', labelsize=font_size, labelcolor=text_color, color = text_color, width=ax_lw)

    ax.grid(False)

    for side in ['bottom', 'left']:
        ax.spines[side].set_linewidth(ax_lw)
        ax.spines[side].set_color(text_color)

    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

    if legend:
        ax.legend(fontsize=legend_font_size, labelcolor=text_color, facecolor=bg_color, edgecolor=text_color)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=x_tick_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.show()