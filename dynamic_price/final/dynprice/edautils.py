import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from typing import Tuple, List, Dict

def get_sku_pivot(dftr: pd.DataFrame, random_count = 25) -> Tuple[pd.DataFrame]:
    """
    dftr contains [SKU, dates, price, user_id]
    """
    assert random_count > 0 or random_count == -1
    if random_count > 0:
        sku_sample = np.random.choice(dftr.SKU.unique(), random_count)
        dfsample = dftr[dftr.SKU.isin(sku_sample)]
    elif random_count == -1:
        dfsample = dftr
    dfsample_agg = dfsample.groupby(by=['dates','SKU','price']).user_id.count().reset_index().rename(columns={'user_id': 'num_purchases'})
    pivot_t = dfsample_agg.pivot_table(index='dates',columns='SKU', values='price')
    return pivot_t, dfsample_agg

def sku_corr_plot(pivot_t: pd.DataFrame) -> None:
    """
    Sequence plot and correlation headmap
    """
    plt.figure(figsize=(12,8))
    sns.lineplot(data = pivot_t, dashes=False)
    plt.show()

    f, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(pivot_t.corr(), dtype=bool))
    sns.heatmap(pivot_t.corr(), mask=mask)
    plt.show()

def top_corr_pairs(pivot_t: pd.DataFrame()) -> pd.DataFrame():
    pairs = pivot_t.corr().abs().unstack().sort_values()
    pairs.index = pairs.index.set_names(['SKU_1', 'SKU_2'])
    pairs = pairs.reset_index().rename(columns = {0:'correlation'})
    high_corr_pairs = pairs[(pairs.correlation >= 0.75) & (pairs.correlation < 0.99)].reset_index(drop=True)
    # hack drop doublicates 0_o
    dd_high_corr_pairs = high_corr_pairs.sort_values(['SKU_1','SKU_2']).drop_duplicates(subset='correlation').reset_index(drop=True)
    assert dd_high_corr_pairs.shape[0] == high_corr_pairs.shape[0] // 2
    print("Shape", dd_high_corr_pairs.shape)
    return dd_high_corr_pairs

def group_corr_chain(dd_high_corr_pairs: pd.DataFrame()) -> Dict:
    chain = {}
    for index, row in dd_high_corr_pairs.iterrows():
        sku_1 = int(row.SKU_1)
        sku_2 = int(row.SKU_2)
        if len([1 for sets in chain.values() if sku_2 in sets]) > 0:
            # print('here and sku:',sku_2)
            # print(chain)
            continue
        if chain.get(sku_1):
            chain[sku_1].add(sku_2)
        else:
            chain[sku_1] = {sku_2}
    return chain

def extend_date(df: pd.DataFrame()) -> pd.DataFrame:
    df = df.copy()
    df.dates = pd.to_datetime(df.dates)
    df['year'] = df.dates.dt.year
    df['month'] = df.dates.dt.month
    df['week_day'] = df.dates.dt.weekday
    df['week_num'] = df.dates.dt.isocalendar().week
    df['dates_int'] = df.dates.view(int)
    return df

def extend_transactions_cols(dftr: pd.DataFrame) -> pd.DataFrame():
    dftr = dftr.copy()
    dftr = dftr.sort_values(['dates', 'SKU'])
    dftr = extend_date(dftr)
    # Среднемесячная цена по SKU
    tr_agg = dftr.groupby(['SKU','year', 'month']).price.mean().reset_index()
    tr_agg['prev_price'] = tr_agg.groupby('SKU').price.shift(1)
    tr_agg['perc_diff'] = tr_agg['prev_price']/tr_agg['price']
    tr_agg['trend_coef'] = np.abs(1 - tr_agg['perc_diff'])
    
    return tr_agg
