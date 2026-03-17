import numpy as np
from sklearn.compose import TransformedTargetRegressor


def create_transformed_target_regressor(regressor):
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)