# -*- coding: utf-8 -*-
import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    """
    Целевые метки, на которое обучается каждое дерево: вместо типичных
    для бустинга ошибок (невязок) используются Lambda-значения.
    """

    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        """
        subsample - доля объектов от выборки, на которых обучается каждое дерево
                    (доля одинакова для всех деревьев, но сама подвыборка генерируется
                    на каждом шагеотдельно)
        colsample_bytree - доля признаков от выборки, на которых обучается каждое дерево
                    (доля одинакова для всех деревьев, но сама подвыборка генерируется на
                    каждом шаге отдельно)
        ps: Совокупность двух параметров выше позволяет реализовать метод случайных подпространств
        Для применения деревьев нужно хранить индексы использованных признаков.
        """
        self._prepare_data()

        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample_count = math.ceil(subsample * self.X_train.shape[0])
        self.colsample_bytree_count = math.ceil(colsample_bytree * self.X_train.shape[-1])
        self.ndcg_top_k = ndcg_top_k

        self.trees = []
        self.tree_col_indices = []
        self.best_ndcg = 0

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
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        # N∗1
        self.ys_train = torch.FloatTensor(y_train).reshape((-1, 1))
        self.ys_test = torch.FloatTensor(y_test).reshape((-1, 1))

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray, inp_query_ids: np.ndarray) -> np.ndarray:
        x = inp_feat_array.copy()
        for qid in np.unique(inp_query_ids):
            idx = inp_query_ids == qid
            x[idx] = StandardScaler().fit_transform(inp_feat_array[idx])
        return x

    def _train_one_tree(self, cur_tree_idx: int, train_preds: torch.FloatTensor) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        """
        cur_tree_idx - номер текущего дерева, в качестве random_seed
        train_preds - суммарные предсказания всех предыдущих деревьев (для расчёта лямбд)

          Рассчитываются лямбды для каждой группы в тренировочном наборе даных, затем 
        применяется метод случайных подпространств, сделав срез по признакам (случайно
        выбранная группа с размером colsample_bytree) и по объектам (случайная группа,
        размер зависит от параметра subsample).
          Затем произвести тренировку одного DecisionTreeRegressor.

        Returns: само дерево и индексы признаков, на которых обучалось дерево.
        """

        lambdas_train = torch.zeros_like(self.ys_train)
        for qid in np.unique(self.query_ids_train):
            idx = self.query_ids_train == qid
            train_preds_slice = train_preds[idx]
            ys_train_slice = self.ys_train[idx]
            lambdas_train[idx] = self._compute_lambdas(ys_train_slice, train_preds_slice)
        indices_col = random.sample(range(self.X_train.shape[-1]), self.colsample_bytree_count)
        indices_col = torch.tensor(indices_col)
        indices_row = random.sample(range(self.X_train.shape[0]), self.subsample_count)
        indices_row = torch.tensor(indices_row)

        X_train_sub = self.X_train.index_select(0, indices_row).index_select(1, indices_col)
        lambdas_train_sub = lambdas_train.index_select(0, indices_row)

        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=cur_tree_idx)
        tree.fit(X_train_sub, -lambdas_train_sub)
        return tree, indices_col.numpy()

    def fit(self):
        """
          Изначальные предсказания до обучения предлагается приравнять к нулю,
        и от этих значений отталкиваться при обучении первого дерева.
        Все обученные деревья необходимо сохранить в список, хранящийся в аттрибуте trees

          Для простоты и ускорения работы предлагается рассчитывать предсказания для всех
        тринировочных и валидационных данных после обучения каждого дерева
        (но досчитывать только изменения за последнее дерево, храня в памяти
        предсказания всех предыдущих вместе).

          Следите за лучшим значением NDCG (хранить в переменной best_ndcg)- после
        окончания тренировки нужно обрезать те последние N деревьев, которые лишь
        ухудшают на валидации метрику (например, вы обучили 100 деревьев, и лучший
        результат был достигнут на 78. Тогда self.trees нужно обрезать до 78 дерева,
        чтобы модель при предсказании работала лучше всего)
        """
        np.random.seed(0)
        self.trees = []
        self.tree_col_indices = []
        self.best_ndcg = 0
        prune_idx = 0
        train_preds = torch.zeros_like(self.ys_train)
        test_preds = torch.zeros_like(self.ys_test)
        for i in tqdm(range(self.n_estimators), total=self.n_estimators):
            tree, col_indices = self._train_one_tree(i, train_preds)
            ys_pred = tree.predict(self.X_train[:, col_indices])
            train_preds += self.lr * ys_pred.reshape((-1, 1))
            self.trees.append(tree)
            self.tree_col_indices.append(col_indices)

            test_preds += tree.predict(self.X_test[:, col_indices]).reshape((-1, 1)) * self.lr
            eval_metric = self._calc_data_ndcg(self.query_ids_test, self.ys_test, test_preds)
            if eval_metric > self.best_ndcg:
                self.best_ndcg = eval_metric
                prune_idx = i+1

            # if i % 10 == 0:
            #     print(eval_metric)
        self.trees = self.trees[:prune_idx]
        self.tree_col_indices = self.tree_col_indices[:prune_idx]

    def eval_test_set(self) -> float:
        return self._calc_data_ndcg(self.query_ids_test, self.ys_test, self.predict(self.X_test))

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        """
        Input: NxD
        """
        results = torch.zeros((data.shape[0], 1), dtype=torch.float32)
        for tree, col_indices in zip(self.trees, self.tree_col_indices):
            results += tree.predict(data[:, col_indices]).reshape((-1, 1)) * self.lr
        return results

    ########
    # Lambda
    ########

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # рассчитаем нормировку, IdealDCG
        ideal_dcg = self._dcg_k(y_true.squeeze(), y_true.squeeze())
        if ideal_dcg == 0:
            return torch.zeros_like(y_true)
        N = 1 / ideal_dcg

        # рассчитаем порядок документов согласно оценкам релевантности
        _, rank_order = torch.sort(y_true, descending=True, axis=0)
        rank_order += 1
        with torch.no_grad():
            # получаем все попарные разницы скоров в батче
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            # поставим разметку для пар, 1 если первый документ релевантнее
            # -1 если второй документ релевантнее
            Sij = self._compute_labels_in_batch(y_true)
            # посчитаем изменение gain из-за перестановок
            gain_diff = self._compute_gain_diff(y_true)

            # посчитаем изменение знаменателей-дискаунтеров
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # посчитаем непосредственное изменение nDCG
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # посчитаем лямбды
            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True, dtype=torch.float32)

            return lambda_update

    def _compute_labels_in_batch(self, y_true: torch.FloatTensor):

        # разница релевантностей каждого с каждым объектом
        rel_diff = y_true - y_true.t()

        # 1 в этой матрице - объект более релевантен
        pos_pairs = (rel_diff > 0).type(torch.float32)

        # 1 тут - объект менее релевантен
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij

    def _compute_gain_diff(self, y_true: torch.FloatTensor, gain_scheme='exp2'):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "const":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff

    #########
    # Metrics
    #########

    def _calc_data_ndcg(self, queries_list: np.ndarray, true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        ndcgs = []
        for qid in np.unique(queries_list):
            idx = queries_list == qid
            batch_ys = true_labels[idx]
            batch_pred = preds[idx]
            val = self._ndcg_k(batch_ys.squeeze(), batch_pred.squeeze(), self.ndcg_top_k)
            ndcgs.append(val)
        return np.mean(ndcgs)

    def _compute_gain(self, ys_value: float, gain_scheme: str = 'exp2') -> float:
        if gain_scheme == "const":
            return ys_value
        elif gain_scheme == "exp2":
            return 2 ** ys_value - 1

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, top_k=0) -> float:
        _, idx = torch.sort(ys_pred, descending=True)
        ys_true = ys_true[idx]
        if top_k > 0:
            ys_true = ys_true[:top_k]
        gain = self._compute_gain(ys_true)
        discount = [math.log2(float(x) + 1) for x in range(1, len(ys_true) + 1)]
        discount = torch.FloatTensor(discount)
        discounted_gain = (gain / discount).sum()
        return discounted_gain.item()

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k: int) -> float:
        current_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        if ideal_dcg > 0:
            return current_dcg / ideal_dcg
        else:
            return 0

    def save_model(self, path: str):
        state = {'trees': self.trees, 'tree_col_indices': self.tree_col_indices,
                 'lr': self.lr, 'best_ndcg': self.best_ndcg}
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.trees = state['trees']
        self.tree_col_indices = state['tree_col_indices']
        self.lr = state['lr']
        self.best_ndcg = state['best_ndcg']


def hyper_optimization():
    from hyperopt import STATUS_OK, Trials, fmin, hp, rand, tpe
    #зададим пространство поиска
    space = [hp.randint('n_estimators', 100, 131),
                hp.uniform('lr', 0.05, 0.5),
                hp.uniform('subsample', 0.4, 0.9),
                hp.uniform('colsample_bytree', 0.65, 0.95),
                hp.randint('max_depth', 2, 6),
                hp.randint('min_samples_leaf', 5, 10),
                ]

    #укажем objective-функцию
    def f(args):
        n_estimators, lr, subsample, colsample_bytree, max_depth, min_samples_leaf = args
        s = Solution(n_estimators = n_estimators, lr =lr,
                 subsample = subsample, colsample_bytree = colsample_bytree,
                 max_depth = max_depth, min_samples_leaf = min_samples_leaf)
        s.fit()
        print(f"Best ndcg: {s.best_ndcg}, tree count = {len(s.trees)} on kwargs: {args}")
        metric = s.eval_test_set()
        if metric >= 0.431:
            s.save_model(f"gbm_lambda_best_{metric:.3}.pkl")
        return -s.eval_test_set()

    best = fmin(f, space, algo = tpe.suggest, max_evals=20)
    print ('TPE result: ', best)

if __name__ == "__main__":
    """
      Предобученная модель показывает NDCG≥0.431.
    можно использовать hyperopt  для подбора параметров

    за 100 деревьев и не более чем за 5 минут с нуля обучаться до NDCG=0.405
    """
    s = Solution()
    s.fit()
    print(f"Best ndcg: {s.best_ndcg}, tree count = {len(s.trees)}")
    s.save_model('gbm_lambda.pkl')
    s.load_model('gbm_lambda.pkl')
    print(s.eval_test_set())
