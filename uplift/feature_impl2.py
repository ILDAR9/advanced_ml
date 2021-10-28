import dask.dataframe as dd
import pandas as pd
import datetime
import featurelib as fl

import category_encoders as ce
import sklearn.base as skbase
import numpy as np


class DayOfWeekReceiptsCalcer(fl.DateFeatureCalcer):
    name = 'day_of_week_receipts'
    keys = ['client_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        purchases = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (purchases['transaction_datetime'] >= date_from) & (purchases['transaction_datetime'] < date_to)

        purchases = purchases.loc[date_mask]
        purchases['day_of_week'] = purchases['transaction_datetime'].dt.weekday
        purchases = purchases.categorize(columns=['day_of_week'])
        features = purchases.pivot_table(
            index='client_id', columns='day_of_week', values='transaction_id', aggfunc='count'
        )

        for day in range(7):
            if day not in features.columns:
                features[day] = 0

        features = features.rename(columns={day: f'purchases_count_dw{day}__{self.delta}d' for day in features.columns}).reset_index()

        return features


class FavouriteStoreCalcer(fl.DateFeatureCalcer):
    name = 'favourite_store'
    keys = ['client_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

        receipts = receipts.loc[date_mask]

        trcount_per_store = (
            receipts
            .groupby(by=['client_id', 'store_id'])
            ['transaction_id'].count()
            .reset_index()
            .rename(columns={"transaction_id": "transaction_count"})
        )

        max_trcount_per_client = (
            trcount_per_store
            .groupby(by=['client_id'])
            ['transaction_count'].max()
            .reset_index()
        )

        # Если таких несколько, выбирать магазин с максимальным номером.
        features = (
            max_trcount_per_client
            .merge(
                trcount_per_store,
                on=['client_id', 'transaction_count']
            )
            .groupby(by=['client_id'])
            ['store_id'].max()
            .reset_index()
            .rename(columns={'store_id': f'favourite_store_id__{self.delta}d'})
        )

        return features


@fl.functional_transformer
def ExpressionTransformer(data: pd.DataFrame, expression: str, col_result: str) -> pd.DataFrame:
    data[col_result] = eval(expression.format(d='data'))
    return data


class LOOMeanTargetEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, col_categorical: str, col_target: str, col_result: str, **loo_params):
        self.col_categorical = col_categorical
        self.col_target = col_target
        self.col_result = col_result
        self.encoder_ = ce.LeaveOneOutEncoder(cols=[col_categorical], **(loo_params or {}))

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        y = None
        if self.col_target in data.columns:
            y = data[self.col_target]
        # ToDo may be exception
        self.encoder_.fit(data[self.col_categorical], y=y)
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        y = None
        if self.col_target in data.columns:
            y = data[self.col_target]
        data[self.col_result] = self.encoder_.transform(data[self.col_categorical], y=y)
        return data


def prepare_engine(fld_store: str) -> fl.Engine:
    receipts = dd.read_parquet(fld_store + 'receipts.parquet')
    campaigns = dd.read_csv(fld_store + 'campaigns.csv')
    client_profile = dd.read_csv(fld_store + 'client_profile.csv')
    products = dd.read_csv(fld_store + 'products.csv')
    purchases = dd.read_parquet(fld_store + 'purchases.parquet')

    TABLES = {
        'receipts': receipts,
        'campaigns': campaigns,
        'client_profile': client_profile,
        'products': products,
        'purchases': purchases,
    }

    engine = fl.Engine(tables=TABLES)
    return engine


def main():
    from data_config import data_config
    from feature_impl import AgeGenderCalcer, TargetFromCampaignsCalcer
    FLD_STORE = "uplift/data/hw4/"
    engine = prepare_engine(fld_store=FLD_STORE)
    test_df = pd.read_csv(FLD_STORE + 'result_dataset_mini.csv')
    SEP = '\n' + "-"*100 + '\n'

    print('\t===Calcers===')
    # Check each calcer alone
    fl.register_calcer(DayOfWeekReceiptsCalcer)
    fl.register_calcer(FavouriteStoreCalcer)

    for cfg in data_config['calcers']:
        name, kwargs = cfg['name'], cfg['args']
        if fl.CALCER_REFERENCE.get(name) is None:
            continue
        calcer = fl.CALCER_REFERENCE[name](engine=engine, **kwargs)
        result_df = calcer.compute().compute()

        assert np.allclose(result_df.values, test_df[result_df.columns].values, atol=1e-4)

        print(f'{name}: {kwargs}')
        print(SEP, result_df.head(), SEP)
    del name, kwargs
    # Prepare all firstline features
    fl.register_calcer(AgeGenderCalcer)
    fl.register_calcer(TargetFromCampaignsCalcer)
    feat_df = fl.compute_features(engine=engine, features_config=data_config['calcers']).compute()

    # Check each transformer alone
    print('\t===Transformers===')
    fl.register_transformer(LOOMeanTargetEncoder, name='loo_mean_target_encoder')
    fl.register_transformer(ExpressionTransformer, name='expression')
    for cfg in data_config['transforms']:
        name = cfg['name']
        cols = [cfg['args']['col_result']]
        if fl.TRANSFORMER_REFERENCE.get(name) is None:
            continue

        transform_config = [cfg]
        pipeline = fl.build_pipeline(transform_config = transform_config)
        result_df = pipeline.fit_transform(feat_df)
        assert np.allclose(result_df[cols].values, test_df[cols].values, atol=1e-4)

        print(name)
        print(SEP, result_df[cols].head(), SEP)


if __name__ == "__main__":
    main()
