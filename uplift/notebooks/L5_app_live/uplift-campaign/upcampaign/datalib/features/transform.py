import pandas as pd
import numpy as np
import sklearn.base as skbase
import category_encoders as ce
import sklearn.preprocessing as skpreprocessing

from typing import List, Optional

from .base import functional_transformer


def divide_cols(data: pd.DataFrame, col_numerator: str, col_denominator: str, col_result: str = None):
    col_result = col_result or f'ratio__{col_numerator}__{col_denominator}'
    data[col_result] = data[col_numerator] / data[col_denominator]
    return data

DivideColsTransformer = functional_transformer(divide_cols)


def do_binning(
    data: pd.DataFrame,
    col_value: str,
    col_result: str,
    bins: List[float],
    labels: Optional[List[str]] = None,
    use_tails: bool = True
) -> pd.DataFrame:
    if use_tails:
        if bins[0] != -np.inf:
            bins = [-np.inf] + bins
        if bins[-1] != np.inf:
            bins = bins + [np.inf]
    data[col_result] = pd.cut(data[col_value], bins=bins, labels=labels)
    return data

BinningTransformer = functional_transformer(do_binning)


class OneHotEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, cols: List[str], prefix: str = 'ohe', **ohe_params):
        self.cols = cols
        self.prefix = prefix
        self.encoder_ = skpreprocessing.OneHotEncoder(**(ohe_params or {}))

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        self.encoder_.fit(data[self.cols])
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        result_column_names = []
        for col_idx, col in enumerate(self.cols):
            result_column_names += [
                f'{self.prefix}__{col}__{value}'
                for i, value in enumerate(self.encoder_.categories_[col_idx])
                if self.encoder_.drop_idx_ is None or i != self.encoder_.drop_idx_[col_idx]
            ]

        encoded = pd.DataFrame(
            self.encoder_.transform(data[self.cols]).todense(),
            columns=result_column_names
        )

        for col in encoded.columns:
            data[col] = encoded[col]
        return data


def expression_transformer(data: pd.DataFrame, expression: str, col_result: str) -> pd.DataFrame:
    data[col_result] = eval(expression.format(d='data'))
    return data

ExpressionTransformer = functional_transformer(expression_transformer)
