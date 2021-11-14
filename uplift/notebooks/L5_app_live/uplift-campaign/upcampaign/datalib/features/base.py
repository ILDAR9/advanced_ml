import sklearn.base as skbase
import sklearn.pipeline as skpipe
import functools
import dask.dataframe as dd
import datetime

from abc import ABC, abstractmethod

from ..connection import Engine


class FeatureCalcer(ABC):
    name = '_base'
    keys = None

    def __init__(self, engine: Engine):
        self.engine = engine

    @abstractmethod
    def compute(self) -> dd.DataFrame:
        pass


class DateFeatureCalcer(FeatureCalcer):
    def __init__(self, date_to: datetime.date, **kwargs):
        self.date_to = date_to
        super().__init__(**kwargs)


class FunctionalTransformer(skbase.BaseEstimator, skbase.TransformerMixin):
    def __init__(self, function, **params):
        self.function = functools.partial(function, **params)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def functional_transformer(function):
    def builder(**params):
        return FunctionalTransformer(function, **params)
    return builder
