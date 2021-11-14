import pandas as pd

from .model_utils import ModelKeeper
from .datalib.features.base import functional_transformer


def apply_model(data: pd.DataFrame, model: ModelKeeper, col_result: str):
    data[col_result] = model.predict(data)
    return data
    
ModelApplyTransformer = functional_transformer(apply_model)
