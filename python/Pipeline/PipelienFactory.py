from ruhken_utils import FeatureDropTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from Pipeline.Decomposer.MSTLDecomposer import MSTLDecomposer, ResidualOptions
from Pipeline.Transformers.ColumnFunctionTransformer import ColumnFunctionTransformer
from Pipeline.Transformers.DropHeadNaN import DropHeadNaN
from Pipeline.Transformers.TemporalFeatureBuilder import TemporalFeatureBuilder


class PipelineFactory:
    @staticmethod
    def create_main_pipeline(pm10_params = None, wind_params = None):
        pm10_pipeline, _ = PipelineFactory.create_pm_pipeline(pm10_params)
        wind_pipeline, _ = PipelineFactory.create_windpipe(wind_params)

        main_pipeline = Pipeline([
            ### Airpolution 
            ("pm10_pipeline", pm10_pipeline),

            ## Weather
            ("wind_pipe", wind_pipeline),
            ("drop_unused_weather_features", FeatureDropTransformer(['precipitation', 'pressure_msl', 'sunshine', 'temperature',
            'cloud_cover', 'dew_point', 'relative_humidity', 'visibility', 'solar']) ),

            ## Put as last pipeline
            ("drop_head_nan", DropHeadNaN()),  
   
            
        ])

        return main_pipeline

    @staticmethod
    def create_pm_pipeline(params: dict = None):
        if params is None:
            params = {
                "seasonal_periods": [7, 365],
                "trend_model": LinearRegression(),
                "pm10_lags": [1, 2, 3, 4, 7, 8, 14, 15, 21, 22, 28, 29],
                "pm10_rolling_windows": [2, 3, 4],
                "residual_lags": [1, 7, 8, 14, 15, 21, 22, 28, 29],
                "residual_rolling_windows": [2, 3, 4],
            }

        pipeline = Pipeline([
            ("log1p_transform_PM10", ColumnFunctionTransformer(["PM10"])),
            ("mstl_decomposition", MSTLDecomposer(column_name="PM10", seasonal_periods=params["seasonal_periods"], trend_model=params["trend_model"], residual_options=ResidualOptions.NEW_FEATURE)),
            ("temporal_feature_builder", TemporalFeatureBuilder(["PM10"], lags=params["pm10_lags"], rolling_windows=params["pm10_rolling_windows"],)),
            ("residual_temporal_feature_builder", TemporalFeatureBuilder(["PM10_residuals"], lags=params["residual_lags"], rolling_windows=params["residual_rolling_windows"])),
            ("drop_PM_10", FeatureDropTransformer(["PM10_residuals", "PM10"])),
        ])


        return pipeline, params
    
    @staticmethod

    def create_windpipe(params: dict = None):
        if params is None:
            params = {
                "seasonal_periods": [365],
                "trend_model": LinearRegression(),
                "lags": [1, 2],
                "rolling_windows": [2, 3, 4],
            }
        
        pipeline = Pipeline([
            #("mstl_decomposer", MSTLDecomposer(column_name='sunshine', seasonal_periods=params["seasonal_periods"], trend_model= params["trend_model"])),
            ("temporal_features", TemporalFeatureBuilder(['w_x', 'w_y', 'wg_x', 'wg_y'], lags=params["lags"], rolling_windows=params["rolling_windows"],  drop_original=True)),
        ])

        return pipeline, params