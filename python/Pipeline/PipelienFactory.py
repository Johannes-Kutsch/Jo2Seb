from ruhken_utils import FeatureDropTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from Pipeline.Decomposer.MSTLDecomposer import MSTLDecomposer
from Pipeline.Transformers.ColumnFunctionTransformer import ColumnFunctionTransformer
from Pipeline.Transformers.DropHeadNaN import DropHeadNaN
from Pipeline.Transformers.TemporalFeatureBuilder import TemporalFeatureBuilder


class PipelineFactory:
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
            ("mstl_decomposition", MSTLDecomposer(column_name="PM10", seasonal_periods=params["seasonal_periods"],
                                                  trend_model=params["trend_model"],
                                                  overwrite_column_with_residuals=False)),
            ("temporal_feature_builder", TemporalFeatureBuilder("PM10", lags=params["pm10_lags"], rolling_windows=params["pm10_rolling_windows"],)),
            ("residual_temporal_feature_builder", TemporalFeatureBuilder("PM10_residuals", lags=params["residual_lags"], rolling_windows=params["residual_rolling_windows"])),
            ("drop_PM_10", FeatureDropTransformer(["PM10_residuals", "PM10"])),
            ("drop_head_nan", DropHeadNaN())
        ])

        return pipeline, params