# -*- coding: utf-8 -*-
"""
Uplift-дерево с критерием разбиения DeltaDeltaP
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Node:
    """
    Параметры взяты из example_tree
    """
    ate: float = None
    n_items: int = None
    split_feat: Any = None
    split_feat_idx: int = None
    split_threshold: float = None
    left = None
    right = None


class UpliftTreeRegressor:

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 1000,
        min_samples_leaf_treated: int = 300,
        min_samples_leaf_control: int = 300
    ):
        """
        Uplift-tree with criterion 'DeltaDeltaP'
        Parameters
        ----------
        max_depth : максимальная глубина дерева.
        min_samples_leaf : минимальное необходимое число обучающих объектов в листе дерева.
        min_samples_leaf_treated : минимальное необходимое число обучающих объектов с T=1 в листе дерева.
        min_samples_leaf_control : минимальное необходимое число обучающих объектов с T=0 в листе дерева.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self._root: Node = None

    def get_root(self) -> Optional[Node]:
        return self._root

    ##########
    # Training
    ##########

    def fit(self, X, treatment, y) -> None:
        """
        Parameters
        ----------
        x : массив (n * k) с признаками.
        treatment : массив (n) с флагом воздействия.
        y : массив (n) с целевой переменной.
        """
        df, self.feat_columns = self._prepare_dataframe(X, treatment, y)
        self.n_features_ = len(self.feat_columns)

        self._root = self._build_tree(df)

    def _prepare_dataframe(self, X, treatment, y) -> Tuple[pd.DataFrame, List]:
        """
        Returns
        -------
        dataframe & columns
        """
        feat_columns = [f'feat{idx}' for idx in range(X.shape[1])]
        df = pd.DataFrame(data=X, columns=feat_columns)
        df['treatment'] = treatment
        df['y'] = y
        return df, feat_columns

    def _build_tree(self, df: pd.DataFrame, depth: int = 0) -> Node:
        '''Build tree
        Pseudo Code
        -----------
        Build(вершина):
            Если вершина на максимальной глубине, то останавливаемся.

            Перебираем все признаки:
                Перебираем “возможные” (определение см. дальше в задаче) варианты порога:
                    Разделяем данные по (признак, порог) -> данные_слева, данные_справа
                    Если разбиение не удовлетворяет условиям на минимальное количество объектов:
                        Continue # т.е. Не рассматриваем это разбиение
                    Считаем критерий delta_delta_p

            Выбираем наилучшее по значению критерия разбиение.
            По лучшему разбиению создаем левую вершину и правую вершину.
            Build(левая вершина)
            Build(правая вершина)
        Parameters
        ----------
        df : Датафрейм со всеми данными.
        depth : node depth.
        Returns
        -------
        Node
            Subtree root node
        '''
        node = Node(
            ate=self._compute_ate(df),
            n_items=df.shape[0]
        )

        if depth >= self.max_depth:
            return node

        best_feat, best_threshold = self._find_split(df)
        if best_feat is not None:
            data_left, data_right = self.split_dataset(df, best_feat, best_threshold)
            node.split_feat = best_feat
            node.split_feat_idx = self.feat_columns.index(best_feat)
            node.split_threshold = best_threshold
            node.left = self._build_tree(data_left, depth + 1)
            node.right = self._build_tree(data_right, depth + 1)

        return node

    def _find_split(self, df: pd.DataFrame) -> Tuple[Any, float]:
        '''
        Разбиения данных для Node по критерию (criterion).
        Parameters
        ----------
            df: датафрейм со всеми признаками.
        Returns
        -------
        Наименование признака & threshold.
        '''
        best_gain = 0.0
        best_feat, best_threshold = None, None

        for feat_name in self.feat_columns:

            column_values = df.loc[:, feat_name]
            unique_values = np.unique(column_values)
            if len(unique_values) > 10:
                percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
            else:
                percentiles = np.percentile(unique_values, [10, 50, 90])
            # получившиеся варианты порога. Их и нужно будет перебрать при подборе оптимального порога.
            threshold_options = np.unique(percentiles)

            for threshold in threshold_options:
                df_left, df_right = self.split_dataset(df, feat_name, threshold)

                # check the split validity on min_samples_leaf

                left_treat_one = sum(df_left.treatment == 1)
                right_treat_one = sum(df_right.treatment == 1)
                if (
                    df_left.shape[0] < self.min_samples_leaf
                    or df_right.shape[0] < self.min_samples_leaf
                    or left_treat_one < self.min_samples_leaf_treated
                    or (df_left.shape[0] - left_treat_one) < self.min_samples_leaf_control
                    or right_treat_one < self.min_samples_leaf_treated
                    or (df_right.shape[0] - right_treat_one) < self.min_samples_leaf_control
                ):
                    continue

                gain = self._compute_gain(df_left, df_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_name
                    best_threshold = threshold

        return best_feat, best_threshold

    @staticmethod
    def split_dataset(df: pd.DataFrame, feat_name: str, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''Split dataset.'''
        mask = df.loc[:, feat_name] <= threshold
        return df[mask], df[~mask]

    def _compute_gain(self, data_left: pd.DataFrame, data_right: pd.DataFrame) -> float:
        '''
        Returns
        -------
        DeltaDeltaP
        '''
        ate_left, ate_right = self._compute_ate(data_left), self._compute_ate(data_right)
        gain: float = abs(ate_left - ate_right)
        return gain

    @staticmethod
    def _compute_ate(df: pd.DataFrame) -> float:
        '''
        Returns
        -------
        Average Treatment Effect Y(1)-Y(0)
        '''
        tmp = df.groupby('treatment')['y'].mean()
        ate: float = tmp.loc[1] - tmp.loc[0]
        return ate

    ############
    # Prediction
    ############

    def predict(self, X: np.ndarray) -> Iterable[float]:
        predictions = [self._predict(row) for row in X]
        return predictions

    def _predict(self, feats: np.array) -> float:
        node = self._root
        while node.left:
            if feats[node.split_feat_idx] < node.split_threshold:
                node = node.left
            else:
                node = node.right
        return node.ate


def print_tree(node: Node, node_name: str = 'Root', depth: int = 0):
    print(depth * '\t' + node_name + (' leaf ' if node.left is None else ' ') + str(node))
    print()

    if node.left is not None:
        print_tree(node.left, node_name='Left', depth=depth + 1)
    if node.right is not None:
        print_tree(node.right, node_name='Right', depth=depth + 1)


def _check(model_constructor, model_params: dict, X, treatment, y, X_test, pred_right) -> bool:
    EPS_THRESHOLD = 2
    model = model_constructor(**model_params)
    model.fit(X, treatment, y)
    print_tree(model.get_root())
    pred = np.array(model.predict(X_test)).reshape(len(X_test))
    eps = np.max(np.abs(pred - pred_right))
    passed = eps < EPS_THRESHOLD
    print(f"{'Passed' if passed else 'Failed'}, eps = {eps}")
    mae = np.sum(pred_right - pred)
    print(f"MAE = {mae}")


if __name__ == '__main__':
    # fld_store = "uplift/data/hw2/" # Local
    fld_store = "" # Server test
    X = np.load(fld_store + 'example_X.npy')
    treatment = np.load(fld_store + 'example_treatment.npy')
    y_hat = np.load(fld_store + 'example_y.npy')
    preds_right = np.load(fld_store + 'example_preds.npy')
    model_params = {'max_depth': 3,
                    'min_samples_leaf': 6000,
                    'min_samples_leaf_treated': 2500,
                    'min_samples_leaf_control': 2500}

    X_train, X_test, y_train, _, treatment_train, _, _, preds_test = train_test_split(X, y_hat, treatment,
                                                                                      preds_right, stratify = treatment,
                                                                                      test_size=0.2, random_state=17)

    _check(UpliftTreeRegressor, model_params, X_train, treatment_train, y_train, X_test, preds_test)
