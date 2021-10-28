from typing import Callable, Dict, List, Tuple, Union, Optional
from flask import Flask
from flask import request
import requests
import os

import pandas as pd
import numpy as np

import uuid

"""
Сервис для связки данных дата, SKU, user предложит цену (price).
ps: могут быть новые покупатели.

При попытке посылать запросы чаще rps=1, status_code = 429
"""

class Strategy:
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.n_iters = 0
        self.arms_states = np.zeros(n_arms)
        self.arms_actions = np.zeros(n_arms)
        
    def flush(self):
        self.n_iters = 0
        self.arms_states = np.zeros(self.n_arms)
        self.arms_actions = np.zeros(self.n_arms)
        
    def update_reward(self, arm: int, reward: int):
        self.n_iters += 1
        self.arms_states[arm] += reward
        self.arms_actions[arm] += 1
        
    def choose_arm(self):
        raise NotImplementedError

class UCB1(Strategy):
    
    def choose_arm(self):
        if self.n_iters < self.n_arms:
            return self.n_iters
        else:
            return np.argmax(self.ucb())

    def ucb(self):
        ucb = self.arms_states / self.arms_actions # mean x_j
        ucb += np.sqrt(2 * np.log(self.n_iters) / self.arms_actions) # confidence part
        return ucb

class Thompson(Strategy):
    
    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        
    def update_reward(self, arm: int, reward: int):
        super().update_reward(arm, reward)
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        
    def choose_arm(self):
        prior_values = np.random.beta(self.alpha, self.beta)
        return np.argmax(prior_values)

class BernoulliEnv:
    
    def __init__(self, arms_proba: list):
        self.arms_proba = arms_proba
        
    @property
    def n_arms(self):
        return len(self.arms_proba)
        
    def pull_arm(self, arm_id: int):
        if random.random() < self.arms_proba[arm_id]:
            return 1
        else:
            return 0

class Bandit:
    
    def __init__(self, env: BernoulliEnv, strategy: Strategy):
        self.env = env
        self.strategy = strategy
        
    def action(self):
        arm = self.strategy.choose_arm()
        reward = self.env.pull_arm(arm)
        self.strategy.update_reward(arm, reward)

def get_uuid():
    fpath_uuid = "uuid.txt"
    if os.path.exists(fpath_uuid):
        with open(fpath_uuid, 'r') as f:
            uuid_val = f.read().strip()
    else:
        uuid_val = uuid.uuid4().hex
        with open(fpath_uuid, 'w') as f:
            f.write(uuid_val)
    print(uuid_val)
    return uuid_val

class DataService:
    
    def __init__(self, uuid_val: str, is_debug = False):
        self.url_begin = f'https://lab.karpov.courses/hardml-api/project-1/task/{uuid_val}/begin'
        self.url_data = f'https://lab.karpov.courses/hardml-api/project-1/task/{uuid_val}/data'
        self.url_result = f'https://lab.karpov.courses/hardml-api/project-1/task/{uuid_val}/result'
    
    def start(self) -> None:
        req = requests.post(self.url_begin)
        print(req.json())
    
    def restart(self) -> None:
        self.start()

    def _request_data(self) -> Optional[pd.DataFrame]:
        response = requests.get(self.url_data)
        json_data: Dict = response.json()
        if json_data['status'].strip() == "batch processing finished":
            return None
        del json_data['status']
        df = pd.read_json(json_data)
        return df
    
    def _upload_results(self, df: pd.DataFrame) -> None:
        # ex. row [{"dates":"2019-12-01","SKU":23198,"user_id":16625,"price":1000.0}]
        req = requests.post(self.url_result, data=df.to_json(orient='records'))

    def _get_recent_purchases(self) -> pd.DataFrame:
        response = requests.get(self.url_result)
        df = pd.read_json(response.json())
        return df

    def process(self) -> None:
        df: Optional[pd.DataFrame] = self._request_data()
        if not df:
            # stop process
            return
        # предсказываем цену для батча какой-то моделью и добавляем её в колонку price
        # Some_Estimator.predict() -> df['price']

if __name__ == "__main__":
    uuid_val = get_uuid()
    data_service = DataService(uuid_val, is_debug=True)

