import pandas as pd
import pickle

from typing import Any, List
from .utils.data import load_pickle


class ModelKeeper:

    def __init__(self, model: Any, column_set: List[str]):
        self.model = model
        self.column_set = column_set
 
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(data[self.column_set])[:, 1]
        else:
            return self.model.predict(data[self.column_set])

    def dump(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(
                obj={'model': self.model, 'column_set': self.column_set},
                file=file
            )


def load_model(path: str) -> ModelKeeper:
    obj = load_pickle(path)
    return ModelKeeper(
        model=obj['model'],
        column_set=obj['column_set'],
    )
