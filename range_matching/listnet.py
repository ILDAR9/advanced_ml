# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List

#########
# Metrics
#########


def compute_gain(y_value: float, gain_scheme: str) -> float:
    """
    вспомогательная функция для расчёта DCG  и NDCG, рассчитывающая показатель gain.
    Принимает на вход дополнительный аргумент - указание схемы начисления gain.

    gain_scheme: ['const', 'exp2'],
    где exp2 - (2^r−1), где r - реальная релевантность документа некоторому запросу
    """
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2.**y_value - 1.
    raise NotImplementedError()


def dcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str) -> float:
    _, idxs = torch.sort(ys_pred, descending=True)
    ys_pred = ys_true[idxs]
    return sum(compute_gain(x.item(), gain_scheme) * 1./math.log2(i) for i, x in enumerate(ys_pred, start=2))


def ndcg(ys_true: torch.Tensor, ys_pred: torch.Tensor, gain_scheme: str = 'exp2') -> float:
    ys_ideal, _ = torch.sort(ys_true, descending=True)
    dcg_val = dcg(ys_true, ys_pred, gain_scheme)
    dcg_ideal = dcg(ys_ideal, ys_ideal, gain_scheme) + 10**-8
    if dcg_ideal == 0:
        return 0.
    return dcg_val/dcg_ideal

################
# Loss functions
################


def listnet_ce_loss(y_i, z_i):
    """
    y_i: (n_i, 1) GT
    z_i: (n_i, 1) preds
    """

    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i))


def listnet_kl_loss(y_i, z_i):
    """
    y_i: (n_i, 1) GT
    z_i: (n_i, 1) preds
    """
    P_y_i = torch.softmax(y_i, dim=0)
    P_z_i = torch.softmax(z_i, dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))

##############
# Architecture
##############


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.Dropout(0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits

##############


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 35,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        """
        Отнормировать признаки X_train и X_test в рамках каждого отдельного идентификатора запроса
        """
        # поместить все переменные с признаками и таргетами в тензоры (torch.FloatTensor) NxD (D-features)
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray, inp_query_ids: np.ndarray) -> np.ndarray:
        x = inp_feat_array.copy()
        for qid in np.unique(inp_query_ids):
            idx = inp_query_ids == qid
            x[idx] = StandardScaler().fit_transform(inp_feat_array[idx])
        return x

    def _create_model(self, listnet_num_input_features: int, listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        """
        Тренировка (не более 60 секунд на 5 эпох)        
        """
        fit_metrics = []
        for i in range(1, self.n_epochs+1):
            self._train_one_epoch()
            eval_metric = self._eval_test_set()
            fit_metrics.append(eval_metric)
        return fit_metrics

    def _train_one_epoch(self) -> None:
        """
        Разовый проход по всем группам из тренировочного датасета. Таким образом,
        на каждом шаге обучения ListNet подаются лишь те объекты выборки,что относятся к одному id
        """
        self.model.train()
        for qid in np.unique(self.query_ids_train):
            idx = self.query_ids_train == qid
            # idx_rand = torch.randperm(sum(idx))
            batch_X = self.X_train[idx,:]
            batch_ys = self.ys_train[idx]

            self.optimizer.zero_grad()
            if len(batch_X) > 0:
                batch_pred = self.model(batch_X).squeeze()
                batch_loss = self._calc_loss(batch_ys, batch_pred)
                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

    def _calc_loss(self, batch_ys: torch.FloatTensor, batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        """
        принимает на вход целевые метки и предсказания, и возвращает значение функции потерь.
        """
        return listnet_ce_loss(batch_ys, batch_pred)
        # return listnet_kl_loss(batch_ys, batch_pred)
        

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            for qid in np.unique(self.query_ids_test):
                idx = self.query_ids_test == qid
                count = sum(idx)
                batch_X = self.X_test[idx,:]
                batch_ys = self.ys_test[idx]
                batch_pred = self.model(batch_X).squeeze()
                val = self._ndcg_k(batch_ys, batch_pred, self.ndcg_top_k)
                ndcgs.append(val)
            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        """
        Если NDCG рассчитать невозможно или ошибка, то NDCG=0 (а не пропускается). 
        """
        _, idxs = torch.sort(ys_pred, descending=True)
        ys_true = ys_true[idxs]
        return ndcg(ys_true[:ndcg_top_k], ys_pred[:ndcg_top_k])


if __name__ == "__main__":
    s = Solution()
    fit_metrics = s.fit()
    for i, eval_metric in enumerate(fit_metrics, start=1):
        print(f"{i}: ndcg@10={eval_metric}")
    # Ожидается NDCG≥0.41
    print(f"Final ndcg@10={s._eval_test_set()}")
