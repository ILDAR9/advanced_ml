# -*- coding: utf-8 -*-
from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    """
    функция для расчёта количества неправильно упорядоченных пар
    (корректное упорядочивание - от наибольшего значения в ys_true к меньшему),
    или переставленных пар 
    """
    _, idxs = sort(ys_pred, descending=True)
    ys_pred = ys_true[idxs]
    wrong_count = 0
    for i, x in enumerate(ys_pred):
        for y in ys_pred[i+1:]:
            if x < y:
                wrong_count += 1
    return wrong_count


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


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _, idxs = sort(ys_pred, descending=True)
    ys_pred = ys_true[idxs]
    return sum(compute_gain(x.item(), gain_scheme) * 1./log2(i) for i, x in enumerate(ys_pred, start=2))


def compute_ideal_dcg(ys_true, ndcg_scheme='exp2'):
    ys_ideal, _ = sort(ys_true, descending=True)
    return dcg(ys_ideal, ys_ideal, ndcg_scheme)

def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    ys_ideal, _ = sort(ys_true, descending=True)
    dcg_val = dcg(ys_true, ys_pred, gain_scheme)
    dcg_ideal = dcg(ys_ideal, ys_ideal, gain_scheme)
    return dcg_val/dcg_ideal


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    """
    функция расчёта точности в топ-k позиций для бинарной разметки
    (в ys_true содержатся только нули и единицы).
    Если среди лейблов нет ни одного релевантного документа (единицы), то нужно вернуть -1.

    k: указывающий на то, по какому количеству объектов необходимо произвести расчёт метрики.
    'k' может быть больше количества элементов во входных тензорах.
    """
    norm_count = ys_true.sum().item()
    if norm_count == 0:
        return -1.
    _, idxs = sort(ys_pred, descending=True)
    ys_pred = ys_true[idxs]
    return ys_pred[:k].sum().item() / min(norm_count, min(k, len(ys_pred)))


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    """
    функция для расчёта MRR (без усреднения, т.е. для одного запроса и множества документов).

    ys_true: могут содержаться только нули и максимум одна единица.
    """
    _, idxs = sort(ys_pred, descending=True)
    ys_pred = ys_true[idxs]
    return 1. / (ys_pred.argmax().item() + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    """
    Базовая вероятность просмотреть первый документ в выдаче pLook[0]=1.

    p_break: вероятность прекращения просмотра списка документов в выдаче.
    ys_true: нормированы от 0 до 1 (вероятность удовлетворения запроса пользователя). 
    """
    _, idxs = sort(ys_pred, descending=True)
    ys_pred = ys_true[idxs]
    p_look_prev = 1.
    p_rel_prev = ys_pred[0].item()
    res = p_rel_prev
    for p_rel in ys_pred[1:]:
        p_look = p_look_prev * (1. - p_rel_prev) * (1. - p_break)
        if p_look <= 0:
            break
        res += p_look * p_rel.item()
        p_rel_prev = p_rel.item()
        p_look_prev = p_look
    return res


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    """
    ∑(Recall@k−Recall@[k−1])⋅Precision@k
    функция расчета AP для бинарной разметки.
    Если среди лейблов нет ни одного релевантного документа (единицы), то нужно вернуть -1.

    ys_true: содержатся только нули и единицы.
    """
    if ys_true.sum().item() == 0:
        return -1.
    _, idxs = sort(ys_pred, descending=True)
    ys_pred = ys_true[idxs]

    res = 0.
    count_right = 0.
    for i, y in enumerate(ys_pred, start=1):
        if y.item():
            count_right += 1.
            res += count_right / i
    return res / count_right


def test_ap():
    import torch
    ys_pred = torch.FloatTensor([0.9, 0.85, 0.71, 0.63, 0.47, 0.36, 0.24, 0.16])
    ys_true = torch.FloatTensor([1, 0, 1, 1, 0, 1, 0, 0])
    res = average_precision(ys_true, ys_pred)
    print(res)


def test_dcg(do_normalize=False):
    import torch
    y_true = torch.FloatTensor([3, 2, 1, 1, 3, 1, 2])
    y_pred = torch.FloatTensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

    scheme = 'const'
    # scheme = 'exp2'
    if do_normalize:
        res = ndcg(y_true, y_pred, gain_scheme=scheme)
    else:
        res = dcg(y_true, y_pred, gain_scheme=scheme)
    print(res)


def test_pfound():
    import torch
    n = 7
    ys_true = torch.FloatTensor([0, 0.5, 1, 1, 0, 0.5, 0])
    ys_pred = torch.rand(n, dtype=torch.float64)
    res = p_found(ys_true, ys_pred)
    print(res)


def test_swapped():
    import torch
    n = 7
    ys_true = torch.FloatTensor([0, 0.5, 1, 1, 0, 0.5, 0])

    res = num_swapped_pairs(ys_true, ys_pred)
    print(res)


def test_p_at_k():
    import torch
    k = 6
    ys_pred = torch.FloatTensor([0.9, 0.85, 0.71, 0.63, 0.47, 0.36, 0.24, 0.16])
    ys_true = torch.FloatTensor([1, 0, 0, 1, 0, 1, 0, 0])
    res = precission_at_k(ys_true, ys_pred, k)
    print(res)


def test_reciprocal():
    import torch
    n = 7
    ys_pred = torch.rand(n, dtype=torch.float64)
    ys_true = torch.zeros(n, dtype=torch.float64)
    ys_true[0] = 1.
    res = reciprocal_rank(ys_true, ys_pred)
    print(res)


if __name__ == "__main__":
    # test_ap()
    # test_dcg()
    # test_dcg(do_normalize=True)
    # test_pfound()
    # test_swapped()
    # test_reciprocal()
    test_p_at_k()
