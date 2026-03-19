from ruhken_utils import FeatureDropTransformer
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from Pipeline.Wrapper.AlignedModelWrapper import AlignedModelWrapper
from Pipeline.Wrapper.AlignedResidualWrapper import AlignedResidualWrapper
from Pipeline.Decomposer.MSTLDecomposer import MSTLDecomposer, ResidualOption
from Pipeline.Decomposer.ProphetDecomposer import ProphetDecomposer
from Pipeline.Transformers.ColumnFunctionTransformer import ColumnFunctionTransformer
from Pipeline.Transformers.DropHeadNaN import DropHeadNaN
from Pipeline.Transformers.TemporalFeatureBuilder import TemporalFeatureBuilder
from Pipeline.Wrapper.WrapperOption import WrapperOption


class PipelineFactory:
    @staticmethod
    def create_main_pipeline(pm10_params = None, wind_params = None):
        pm10_pipeline, pm10_params = PipelineFactory.create_prophet_pm10_pipeline(pm10_params)
        wind_pipeline, wind_params = PipelineFactory.create_windpipe(wind_params)

        main_pipeline = Pipeline([
            ### Airpolution 
            ("pm10_pipeline", pm10_pipeline),

            ## Weather
            #("wind_pipe", wind_pipeline),
            ("drop_unused_weather_features", FeatureDropTransformer(['w_x', 'w_y', 'wg_x', 'wg_y', 'precipitation', 'pressure_msl', 'sunshine', 'temperature',
            'cloud_cover', 'dew_point', 'relative_humidity', 'visibility', 'solar']) ),

            ## Put as last pipeline
            ("drop_head_nan", DropHeadNaN()),
        ])

        return main_pipeline, pm10_params, wind_params

    @staticmethod
    def create_model_pipeline(model: RegressorMixin, wrapper_option: WrapperOption = WrapperOption.DEFAULT, pm10_params = None, wind_params = None):
        main_pipeline, pm10_params, wind_params = PipelineFactory.create_main_pipeline(pm10_params ,wind_params)

        model_pipeline = Pipeline([
            ("main_pipeline", main_pipeline),
            ("model", PipelineFactory._create_model_pipeline(model, wrapper_option, pm10_params))
        ])

        return model_pipeline

    @staticmethod
    def create_mstl_pm_pipeline(params: dict = None):
        default_params = {
            "seasonal_periods": [7, 365],
            "trend_model": LinearRegression(),
            "pm10_lags": [1, 7, 8, 14, 15, 21, 22, 28, 29],
            "pm10_rolling_windows": [2, 3, 4],
        }

        params = PipelineFactory._aggregate_params(params, default_params)

        pipeline = Pipeline([
            ("log1p_transform_PM10", ColumnFunctionTransformer(["PM10"])),
            ("mstl_decomposition", MSTLDecomposer(column_name="PM10", seasonal_periods=params["seasonal_periods"], trend_model=params["trend_model"], residual_options=ResidualOption.IGNORE)),
            ("temporal_feature_builder", TemporalFeatureBuilder(["PM10"], lags=params["pm10_lags"], rolling_windows=params["pm10_rolling_windows"])),
            ("drop_PM_10", FeatureDropTransformer(["PM10_residuals", "PM10"])),
        ])

        return pipeline, params

    @staticmethod
    def create_prophet_pm10_pipeline(params: dict = None):
        default_params = {
            "seasonal_periods": [7, 365],
            "trend_model": LinearRegression(),
            "pm10_lags": [1, 7, 8, 14, 15, 21, 22, 28, 29],
            "pm10_rolling_windows": [2, 3, 4],
            "decomposer": ProphetDecomposer("PM10", residual_options=ResidualOption.IGNORE)
        }

        params = PipelineFactory._aggregate_params(params, default_params)

        pipeline = Pipeline([
            ("log1p_transform_PM10", ColumnFunctionTransformer(["PM10"])),
            ("prophet_decomposition", params["decomposer"]),
            ("temporal_feature_builder", TemporalFeatureBuilder(["PM10"], lags=params["pm10_lags"], rolling_windows=params["pm10_rolling_windows"])),
            ("drop_PM_10", FeatureDropTransformer(["PM10"])),
        ])

        return pipeline, params

    @staticmethod
    def create_windpipe(params: dict = None):
        default_params = {
            "seasonal_periods": [365],
            "trend_model": LinearRegression(),
            "lags": [1, 2],
            "rolling_windows": [2, 3, 4],
        }

        params = PipelineFactory._aggregate_params(params, default_params)

        pipeline = Pipeline([
            #("mstl_decomposer", MSTLDecomposer(column_name='sunshine', seasonal_periods=params["seasonal_periods"], trend_model= params["trend_model"])),
            ("temporal_features", TemporalFeatureBuilder(['w_x', 'w_y', 'wg_x', 'wg_y'], lags=params["lags"], rolling_windows=params["rolling_windows"],  drop_original=True)),
        ])

        return pipeline, params

    @staticmethod
    def _create_model_pipeline(model: RegressorMixin, wrapper_option: WrapperOption, pm10_params: dict):
        if wrapper_option == WrapperOption.DEFAULT:
            return AlignedModelWrapper(model=model)

        elif wrapper_option == WrapperOption.RESIDUAL_COMPOSER:
            return AlignedResidualWrapper(model=model, decomposer=pm10_params["decomposer"])

        raise ValueError(f"Unsupported wrapper_option: {wrapper_option}. Must be one of {list(WrapperOption)}")

    @staticmethod
    def _aggregate_params(params: dict | None, default_params: dict) -> dict:
        if params is None:
            params = {}



        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        return params