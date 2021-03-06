METRIC_VALUES = [0 for _ in range(10000000)] # сюда нужно залить предикты модели
# METRIC_INDEX = [0 for _ in range(10000000)]

import os
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from flask import Flask, jsonify, request


METRIC_NAME = 'sales_cuped'

DF_USERS = pd.read_csv(os.environ['PATH_DF_USERS'])
DF_USERS['strat'] = (
    DF_USERS['gender'] * 2
    + (DF_USERS['age'] >= 26).astype(int)
)

METRIC_INDEX = sorted(DF_USERS['user_id'].unique())
DF_METRICS = pd.DataFrame({METRIC_NAME: METRIC_VALUES}, index=METRIC_INDEX)

app = Flask(__name__)


#########    ФУНКЦИИ ДЛЯ ПРОВЕРКИ T-TEST'a   #########

def calc_strat_mean_var(df: pd.DataFrame, strat_column: str, target_name: str, strat_weights: pd.Series):
    """Считаем стратифицированный среднии и дисперсии.
    
    df - датафрейм с целевой метрикой и данными для стратификации
    strat_column - названия столбца по которому проводить стратификацию
    target_name - название столбца с целевой переменной
    weights - словарь - {название страты: вес страты}
    
    return: (float, float), mean_strat, var_strat
    """
    strat_mean = df.groupby(strat_column)[target_name].mean()
    mean = (strat_mean * strat_weights).sum()
    strat_var = df.groupby(strat_column)[target_name].var()
    var = (strat_var * strat_weights).sum()
    return mean, var


def run_strat_ttest(
    df_pilot: pd.DataFrame, df_control: pd.DataFrame,
    strat_column: str, target_name: str,
    strat_weights: pd.Series
):
    """Проверяет гипотезу о равенстве средних.
    
    Возвращает 1, если среднее значение метрики в пилотной группе
    значимо больше контрольной, иначе 0.
    """
    mean_strat_pilot, var_strat_pilot = calc_strat_mean_var(
        df_pilot, strat_column, target_name, strat_weights
    )
    mean_strat_control, var_strat_control = calc_strat_mean_var(
        df_control, strat_column, target_name, strat_weights
    )

    delta_mean_strat = mean_strat_pilot - mean_strat_control
    std_mean_strat = (var_strat_pilot / len(df_pilot) + var_strat_control / len(df_control)) ** 0.5

    left_bound = delta_mean_strat - 1.96 * std_mean_strat
    right_bound = delta_mean_strat + 1.96 * std_mean_strat
    return int(left_bound > 0) 


def check_strat_balans(df_a_one, df_a_two, df_b, min_count=2):
    """Проверяет наличие страт в группах.
    
    df_a_one, df_a_two, df_b - датафреймы групп (index - user_id, columns = ['strat', ...])
    min_count - минимальное кол-во обёектов страты в каждой группе
    
    Возвращает True, если страты представлены во всех группах, иначе False."""
    list_strats = [df['strat'].values for df in [df_a_one, df_a_two, df_b]]
    unique_strats = np.unique(np.hstack(list_strats))
    
    for unique_strat in unique_strats:
        for group_starts in list_strats:
            count = np.sum(group_starts == unique_strat)
            if count < min_count:
                return False
    return True


def _check_test(
    df_metrics: pd.DataFrame, df_users: pd.DataFrame,
    metric_name: str, test: dict
) -> float:
    """Проверяет гопотезу о равенстве средних AA и AB тестов.
    
    df_metrics - датафрейм с метриками (index - user_id, columns - metrics)
    df_users - датафрейм с информацией о пользователях
    metric_name - название столбца с метрикой в df_metrics
    test - информация о пилоте:
        test_id - id теста 
        group_a_one: list, список user_id группы A_one
        group_a_two: list, список user_id группы A_two
        group_b: list, список user_id группы B
    
    return: 1 - если эффект есть, иначе 0.
    """
#     return int(np.random.randint(0, 2))
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']
    user_ids = group_a_one + group_a_two + group_b
    
    df = pd.concat(
        [
            df_metrics[df_metrics.index.isin(user_ids)][metric_name],
            df_users[df_users['user_id'].isin(user_ids)].set_index('user_id')['strat']
        ],
        axis=1
    )
    
    df_a_one = df[df.index.isin(group_a_one)].copy()
    df_a_two = df[df.index.isin(group_a_two)].copy()
    df_b = df[df.index.isin(group_b)].copy()
    
    # если группы не сопоставимы по стратам (в одной есть элементы одной страты, а в другой нет)
    # то не можем оценить такой эксперимент
    if not check_strat_balans(df_a_one, df_a_two, df_b):
        print(f"pilot_id = {test['test_id']} ,imbalance strat")
        return -1, -1

    strat_weights = df['strat'].value_counts() / df['strat'].count()
    
    # AA тест
    res_aa = run_strat_ttest(
        df_pilot=df_a_one,
        df_control=df_a_two,
        strat_column='strat',
        target_name=metric_name,
        strat_weights=strat_weights
    )

    # AB тест
    res_ab = run_strat_ttest(
        df_pilot=df_b,
        df_control=pd.concat((df_a_one, df_a_two)),
        strat_column='strat',
        target_name=metric_name,
        strat_weights=strat_weights
    )
    
    if res_aa != 0:
        return 0
    if res_ab == 1:
        return 1
    return 0


#########    ФУНКЦИИ FLASK   #########

@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(DF_METRICS, DF_USERS, METRIC_NAME, test)
    return jsonify(has_effect=int(has_effect))
