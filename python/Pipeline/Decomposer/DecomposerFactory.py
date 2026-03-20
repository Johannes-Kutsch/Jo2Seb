from sklearn.linear_model import LinearRegression

from Pipeline.Decomposer.DecomposerOption import DecomposerOption
from Pipeline.Decomposer.MSTLDecomposer import MSTLDecomposer
from Pipeline.Decomposer.ProphetDecomposer import ProphetDecomposer
from Pipeline.Decomposer.ResidualOption import ResidualOption


class DecomposerFactory:
    @staticmethod
    def create_mstl_decomposer(column_name: str = "PM10", seasonal_periods: list[int] = None, trend_model=None,
                               residual_options: ResidualOption = ResidualOption.IGNORE, ) -> MSTLDecomposer:
        if seasonal_periods is None:
            seasonal_periods = [7, 365]
        if trend_model is None:
            trend_model = LinearRegression()

        return MSTLDecomposer(column_name=column_name, seasonal_periods=seasonal_periods, trend_model=trend_model,
                              residual_options=residual_options, )

    @staticmethod
    def create_prophet_decomposer(column_name: str = "PM10", residual_options: ResidualOption = ResidualOption.IGNORE,
                                  **prophet_kwargs, ) -> ProphetDecomposer:
        return ProphetDecomposer(column_name=column_name, residual_options=residual_options, **prophet_kwargs, )

    @staticmethod
    def create_decomposer(decomposer_option):
        if decomposer_option == DecomposerOption.PROPHET:
            return DecomposerFactory.create_prophet_decomposer()
        elif decomposer_option == DecomposerOption.MSTL:
            return DecomposerFactory.create_mstl_decomposer()

        raise ValueError(f"Unsupported decomposer_option: {decomposer_option}. Must be one of {list(DecomposerOption)}")
