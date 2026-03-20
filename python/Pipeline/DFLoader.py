import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from Utils import FilePaths

class DFLoader:
    @staticmethod
    def load_combined_df() -> pd.DataFrame:
        pm10_df = DFLoader.load_pm10_df()
        weather_df = DFLoader.load_weather_df()
        df_combined = pd.concat([pm10_df, weather_df], axis=1)
        return df_combined

    @staticmethod
    def load_daily_df() -> pd.DataFrame:
        combined_df = DFLoader.load_combined_df()
        return combined_df.resample("D").mean()

    @staticmethod
    def load_pm10_df() -> pd.DataFrame:
        df = pd.read_csv(FilePaths.UWB_DATA / "Station_857_2018-01-01--2025-12-31.csv", index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        components_meta = pd.read_csv(FilePaths.UWB_METADATA / "UWB_components_metadata.csv")

        components = ["1"]

        component_pipe = Pipeline([
            ("interpolate_columns_and_drop_other_features", ColumnTransformer(
                transformers=[("interpolate_nan", FunctionTransformer(DFLoader._interpolate_linear), components)],
                remainder="drop"
            ))
        ])

        component_columns = [
            DFLoader._get_component_metadata(int(component), components_meta)["component_code"]
            for component in components
        ]

        transformed = component_pipe.fit_transform(df)

        return pd.DataFrame(transformed, columns=component_columns, index=df.index)

    @staticmethod
    def load_weather_df() -> pd.DataFrame:
        df = pd.read_csv(FilePaths.DWD_PROCESSED_DATA / "Weather_hourly_cleaned-Fuhlsbüttel.csv", index_col=0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df

    @staticmethod
    def load_copernicus_df(size = "small") -> pd.DataFrame:
        df = pd.read_csv(FilePaths.COPERNICUS / f"Sentinel-5P NO2-NO2_{size}-area.csv", index_col=0)
        df.index = pd.to_datetime(df.index, format="%m-%d-%y").tz_localize(None)
        return df

    @staticmethod
    def _get_component_metadata(component_id: int, components_meta: pd.DataFrame):
        return components_meta.loc[components_meta["component_id"] == component_id].squeeze()

    @staticmethod
    def _interpolate_linear(X):
        return X.interpolate(method="linear", limit_direction="both")