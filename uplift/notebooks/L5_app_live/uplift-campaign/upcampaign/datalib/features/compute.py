import sklearn.base as skbase
import sklearn.pipeline as skpipe
import functools
import dask.dataframe as dd
import datetime

from typing import List

from . import extract
from . import transform
from .base import FeatureCalcer
from ..connection import Engine


def register_calcer(calcer_class, calcer_reference) -> None:
    calcer_reference[calcer_class.name] = calcer_class


CALCER_REFERENCE = {}
register_calcer(extract.ReceiptsBasicFeatureCalcer, CALCER_REFERENCE)
register_calcer(extract.UniqueCategoriesCalcer, CALCER_REFERENCE)
register_calcer(extract.AgeGenderCalcer, CALCER_REFERENCE)
register_calcer(extract.TargetFromCampaignsCalcer, CALCER_REFERENCE)


def create_calcer(name: str, calcer_reference=CALCER_REFERENCE, **kwargs) -> FeatureCalcer:
    return calcer_reference[name](**kwargs)


def join_tables(tables: List[dd.DataFrame], on: List[str], how: str) -> dd.DataFrame:
    result = tables[0]
    for table in tables[1: ]:
        result = result.merge(table, on=on, how=how)
    return result


def extract_features(engine: Engine, config: dict, calcer_reference=CALCER_REFERENCE) -> dd.DataFrame:
    calcers = list()
    keys = None

    for feature_config in config:
        calcer_args = feature_config["args"]
        calcer_args["engine"] = engine

        calcer = create_calcer(feature_config["name"], calcer_reference=calcer_reference, **calcer_args)
        if keys is None:
            keys = set(calcer.keys)
        elif set(calcer.keys) != keys:
            raise KeyError(f"{calcer.keys}")

        calcers.append(calcer)

    computation_results = []
    for calcer in calcers:
        computation_results.append(calcer.compute())
    result = join_tables(computation_results, on=list(keys), how='outer')

    return result


def register_transformer(transformer_class, name: str, transformer_reference) -> None:
    transformer_reference[name] = transformer_class


TRANSFORMER_REFERENCE = {}
register_transformer(transform.ExpressionTransformer, 'expression', TRANSFORMER_REFERENCE)
register_transformer(transform.BinningTransformer, 'binning', TRANSFORMER_REFERENCE)
register_transformer(transform.OneHotEncoder, 'one_hot_encode', TRANSFORMER_REFERENCE)


def create_transformer(name: str, transformer_reference=TRANSFORMER_REFERENCE, **kwargs) -> skbase.BaseEstimator:
    return transformer_reference[name](**kwargs)


def create_pipeline(transform_config: dict, transformer_reference=TRANSFORMER_REFERENCE) -> skpipe.Pipeline:
    transformers = list()

    for i, transformer_config in enumerate(transform_config):
        transformer_args = transformer_config["args"]

        transformer = create_transformer(transformer_config["name"], transformer_reference=transformer_reference, **transformer_args)
        uname = transformer_config.get("uname", f'stage_{i}')

        transformers.append((uname, transformer))

    pipeline = skpipe.Pipeline(transformers)
    return pipeline
