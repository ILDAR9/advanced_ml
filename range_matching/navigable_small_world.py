# -*- coding: utf-8 -*-
"""
Построение Small World графа и алгоритм навигации по нему Navigable Small World (NSW).
"""

from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """
    Рассчитывает расстояние от точки pointA (вектор размера 1∗D, где D - размерность эмбеддинга)
    до всех объектов во множестве documents (матрица размера N∗D, где N - количество документов).
    Метрика - Евклидова.

    Returns: N*1
    """
    return np.linalg.norm(pointA - documents, axis=1, keepdims=True)


def create_sw_graph(
    data: np.ndarray,
    num_candidates_for_choice_long: int = 10,
    num_edges_long: int = 5,
    num_candidates_for_choice_short: int = 10,
    num_edges_short: int = 5,
    use_sampling: bool = False,
    sampling_share: float = 0.05,
    dist_f: Callable = distance
) -> Dict[int, List[int]]:
    """
    Граф связей в множестве документов data (матрица размеров N∗D).

    num_candidates_for_choice_long - количество кандидатов, являющихся самыми далекими объектами,
    из которого случайно выбирается

    num_edges_long - штук для построения ребер графа. То есть сначала для каждой точки по честному
    рассчитываются самые дальние точки, после чего отбирается топ самых далеких, и уже из их числа
    происходит семплирование.

    num_candidates_for_choice_short и num_edges_short - для коротких ребер аналогичные параметры.

    use_sampling - для расчёта на больших выборках можно использовать семплинг.

    sampling_share - вместо расчёта расстояний до всех точек рассчитывается лишь доля расстояний.
    ex: sampling_share=0.01 для 10 миллионов документов означает, что для каждой точки случайно
    выбирается 10 тысяч точек, и лишь из их числа выбираются самые близкие и дальние).

    Returns: словарь, где ключ - индекс точки (и количество ключей равно количеству точек N),
    а значение - список индексов точек, которыми образованы связи в виде ребер (длинных и коротких
    совместно, без разделения)
    """
    adjacency_list = defaultdict(list)
    data_size = data.shape[0]

    for cur_point_idx in tqdm(range(data_size), total=data_size):
        if use_sampling:
            sample_size = int(data_size * sampling_share)
            idx_sampled = np.random.choice(np.arange(data_size), size=sample_size, replace=False)
            dists = dist_f(data[cur_point_idx, :], data[idx_sampled, :]).squeeze()
            idx_sorted = idx_sampled[np.argsort(dists)[1:]]
        else:
            dists = dist_f(data[cur_point_idx, :], data).squeeze()
            idx_sorted = np.argsort(dists)[1:]

        short_choice = np.random.choice(idx_sorted[:num_candidates_for_choice_short],
                    size=num_edges_short, replace=False)

        long_choice = np.random.choice(idx_sorted[-num_candidates_for_choice_long:],
                    size=num_edges_long, replace=False)

        adjacency_list[cur_point_idx] += short_choice.tolist() + long_choice.tolist()

    return dict(adjacency_list)


def nsw(query_point: np.ndarray, all_documents: np.ndarray,
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    """
    Осуществляет поиск в графе.

    query_point - точка, к которой нужно искать ближайших соседей в SW-графе среди всех документов.

    graph_edges - результат работы метода cerate_sw_graph, то есть указание всех ребер в графе.

    search_k - должно возвращаться не менее search_k объектов.

    num_start_points - количество случайно выбранных стартовых точек для расчётов. Возможно,
    в некоторых случаях потребуется начать с более чем num_start_points точек.

    Returns: список индексов search_k ближайших к query_point объектов среди all_documents.
    ps: размерность возвращаемого значения - K.
    """
    visited: Dict[int, float] = dict()
    started_points_count = 0
    query_dist = lambda p_idx: dist_f(query_point, all_documents[p_idx, :].reshape(1, -1))[0][0]
    
    while ((started_points_count < num_start_points) or (len(visited) < search_k)):
        cur_idx = np.random.randint(all_documents.shape[0])
        if cur_idx in visited:
            continue
        cur_dist = query_dist(cur_idx)
        visited[cur_idx] = cur_dist

        while True:
            candidate = cur_idx
            min_dist = cur_dist

            already_visited = {candidate}
            for cand_idx in graph_edges[candidate]:
                if cand_idx in visited:
                    already_visited.add(cand_idx)
                    cand_dist = visited[cand_idx]
                else:
                    cand_dist = query_dist(cand_idx)
                    visited[cand_idx] = cand_dist
                # update candidate if possible
                if cand_dist < min_dist:
                    min_dist = cand_dist
                    candidate = cand_idx
            
            if candidate in already_visited:
                break
            cur_dist = min_dist
            cur_idx = candidate
            
        started_points_count += 1

    return sorted(visited.keys(), key = visited.get)[:search_k]


if __name__ == "__main__":
    N, K = 1000, 128
    data = np.random.rand(N, K)
    sw_graph = create_sw_graph(data, use_sampling=False, sampling_share=0.05)
    query_point = np.random.rand(1, K)
    search_k = 10
    res = nsw(query_point, all_documents=data, graph_edges=sw_graph, search_k=search_k)
    print(f"Found {search_k} points:", res)
    print(distance(query_point, data[res,:]))

