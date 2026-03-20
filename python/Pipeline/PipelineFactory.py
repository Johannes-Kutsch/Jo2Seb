from ruhken_utils import FeatureDropTransformer
from sklearn.base import RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Pipeline.Decomposer.DecomposerFactory import DecomposerFactory
from Pipeline.Decomposer.DecomposerOption import DecomposerOption
from Pipeline.Imputers.ProphetImputer import ProphetImputer
from Pipeline.ModelWrapper.AlignedYWrapper import AlignedYWrapper
from Pipeline.ModelWrapper.Recomposer.SeasonalTrendRecomposer import SeasonalTrendRecomposer
from Pipeline.Scalers.ColumnScaler import ColumnScaler
from Pipeline.Transformers.ColumnFunctionTransformer import ColumnFunctionTransformer
from Pipeline.Imputers.ColumnImputer import ColumnImputer
from Pipeline.Transformers.DropHeadNaN import DropHeadNaN
from Pipeline.Transformers.TemporalFeatureBuilder import TemporalFeatureBuilder
from Pipeline.ModelWrapper.Recomposer.RecomposerOption import RecomposerOption


class PipelineFactory:
    @staticmethod
    def create_main_pipeline(decomposer_option = DecomposerOption.PROPHET, pm10_params = None):
        pm10_pipeline, pm10_params = PipelineFactory.create_pm10_pipeline(pm10_params, decomposer_option)
        weather_pipe = PipelineFactory.create_weather_pipe()
        no2_pipe = PipelineFactory.create_no2_pipe()

        main_pipeline = Pipeline([
            ### Airpolution 
            ("pm10_pipeline", pm10_pipeline),

            ## Weather
            ("weather_pipe", weather_pipe),
            ("no2_pipe", no2_pipe),

            ## Put as last pipeline
            ("drop_head_nan", DropHeadNaN()),
        ])

        return main_pipeline, pm10_params

    @staticmethod
    def create_model_pipeline(model: RegressorMixin, decomposer_option = DecomposerOption.PROPHET, recomposer_option = RecomposerOption.NONE, pm10_params = None):
        main_pipeline, pm10_params = PipelineFactory.create_main_pipeline(decomposer_option, pm10_params)

        model_pipeline = Pipeline([
            ("main_pipeline", main_pipeline),
            ("model", PipelineFactory._create_model_pipeline(model, recomposer_option, pm10_params))
        ])

        return model_pipeline

    @staticmethod
    def create_pm10_pipeline(params: dict = None, decomposer_option: DecomposerOption = DecomposerOption.PROPHET):
        default_params = {
            "pm10_lags": [1, 7, 8, 14, 15, 21, 22, 28, 29],
            "pm10_rolling_windows": [2, 3, 4],
        }

        params = PipelineFactory._aggregate_params(params, default_params)

        if not params.get("decomposer"):
            params["decomposer"] = DecomposerFactory.create_decomposer(decomposer_option)

        pipeline = Pipeline([
            ("log1p_transform_PM10", ColumnFunctionTransformer(["PM10"])),
            ("robust_scaler", ColumnScaler(columns=["PM10"])),
            ("decomposition", params["decomposer"]),
            ("temporal_feature_builder", TemporalFeatureBuilder(["PM10"], lags=params["pm10_lags"], rolling_windows=params["pm10_rolling_windows"])),
            ("drop_PM_10", FeatureDropTransformer(["PM10"])),
        ])

        return pipeline, params

    @staticmethod
    def create_weather_pipe():
        pipeline = Pipeline([
            ("drop_false_measurements", FeatureDropTransformer(['pressure_msl'])),
            ("impute_wind", ProphetImputer(columns=["w_x", "w_y", "wg_x", "wg_y"], max_gap_warning=30)),
            ("impute_visibility", ColumnImputer(columns=["visibility"], imputer=SimpleImputer(strategy="constant", fill_value=0))),
            ("robust_scaler", ColumnScaler(columns=["sunshine", "precipitation", "visibility", "solar"])),
            ("standard_scaler", ColumnScaler(columns=["temperature", "dew_point", "w_x", "w_y", "wg_x", "wg_y"], scaler=StandardScaler())),
            ("rescale_100_to_1", ColumnFunctionTransformer(["cloud_cover", "relative_humidity"], function=lambda x: x / 100)),
            ("temporal_feature_builder_precipitation", TemporalFeatureBuilder(['precipitation'], lags=[1], rolling_windows=[7])),
            ("temporal_feature_builder_sunshine", TemporalFeatureBuilder(['sunshine'], lags=[1])),
            ("temporal_feature_builder_cloud_cover", TemporalFeatureBuilder(['cloud_cover'], lags=[1, 2], rolling_windows=[3, 7])),
            #("temporal_feature_builder_temperature", TemporalFeatureBuilder(['temperature'], lags=[5], rolling_windows=[20])),
            #("temporal_feature_builder_dew_point", TemporalFeatureBuilder(['dew_point'], lags=[4], rolling_windows=[5, 7])),
            #("temporal_feature_builder_relative_humidity", TemporalFeatureBuilder(['relative_humidity'], lags=[1, 2], rolling_windows=[2, 4])),
            ("temporal_feature_builder_visibility", TemporalFeatureBuilder(['visibility'], lags=[1], rolling_windows=[2, 3])),
            ("temporal_feature_builder_solar", TemporalFeatureBuilder(['solar'], rolling_windows=[60])),
            ("temporal_feature_builder_w_x", TemporalFeatureBuilder(['w_x'], lags=[1, 2], rolling_windows=[2, 3, 5, 90])),
            ("temporal_feature_builder_w_y", TemporalFeatureBuilder(['w_y'], lags=[1, 2], rolling_windows=[2, 3, 5, 90])),
            ("temporal_feature_builder_wg_x", TemporalFeatureBuilder(['wg_x'], lags=[1, 2], rolling_windows=[2, 3, 4, 90])),
            ("temporal_feature_builder_wg_y", TemporalFeatureBuilder(['wg_y'], lags=[1, 2], rolling_windows=[2, 3, 4, 90])),
            ("drop_orig", FeatureDropTransformer(['precipitation', 'sunshine', 'temperature', 'cloud_cover', 'dew_point',
                 'relative_humidity', 'visibility', 'solar', 'w_x', 'w_y', 'wg_x', 'wg_y']))
        ])

        return pipeline

    @staticmethod
    def create_no2_pipe():
        pipeline = Pipeline([
            ("drop_unused_features", FeatureDropTransformer(['C0/min', 'C0/noDataCount', 'C0/p10', 'C0/sampleCount', 'C0/p90', 'C0/median', 'C0/stDev'])),
            # ("log1p_transform", ColumnFunctionTransformer(['C0/mean', 'C0/max'])),
            # ("robust_scaler", ColumnScaler(columns=['C0/mean',  'C0/max'])),
            # ("impute_NaN", ProphetImputer(columns=['C0/mean', 'C0/max'], max_gap_warning=None)),
            # ("temporal_feature_builder_mean", TemporalFeatureBuilder(['C0/mean'], rolling_windows=[2])),
            # ("temporal_feature_builder_max", TemporalFeatureBuilder(['C0/max'], rolling_windows=[2])),
            ("drop_orig", FeatureDropTransformer(['C0/mean', 'C0/max']))
        ])

        return pipeline

    @staticmethod
    def _create_model_pipeline(model: RegressorMixin, wrapper_option: RecomposerOption, pm10_params: dict):
        if wrapper_option == RecomposerOption.NONE:
            return AlignedYWrapper(model=model)

        elif wrapper_option == RecomposerOption.SEASONAL_TREND:
            return SeasonalTrendRecomposer(model=model, decomposer=pm10_params["decomposer"])

        raise ValueError(f"Unsupported wrapper_option: {wrapper_option}. Must be one of {list(RecomposerOption)}")

    @staticmethod
    def _aggregate_params(params: dict | None, default_params: dict) -> dict:
        if params is None:
            params = {}

        for key, value in default_params.items():
            if key not in params:
                params[key] = value
        return params