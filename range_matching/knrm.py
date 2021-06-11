# -*- coding: utf-8 -*-
"""
1. Реализация токенизации и препроцессинга данных.
2. Создание матрицы эмбеддингов и словаря токенов.
3. Имплементация Kernels и модели KNRM
4. Подготовка Datasets и Dataloaders для обучения и валидации модели.
5. Тренировка модели.
"""
import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F


glue_qqp_dir = '/home/nur/projects/analysis/range_matching/data/QQP'
glove_path = '/home/nur/projects/analysis/range_matching/data/glove.6B.50d.txt'

########
# Блок 3
########


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        """
        mu - “среднее” ядра, точку внимания
        sigma - ширина “бина”
        """
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        """
        Простой нелинейный оператор
        """
        return torch.exp(-(x-self.mu)**2 / (2 * self.sigma**2))


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        """
        Формирует список всех ядер (K штук), применяемых в алгоритме. Важно обратить внимание
        на автоматическую генерацию mu для каждого ядра, а также на крайнее правое значение.
        К примеру, если K=5, то mu должны быть [-0.75, -0.25, 0.25, 0.75, 1],
        а для K=11 [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1].
        Параметр exact_sigma означает sigma-значение
        для крайнего бина, в котором mu равняется единице.
        self.kernel_num > 1
        """
        step = 1. / (self.kernel_num-1)
        mlist = np.arange(-1, 1, step)[1::2].tolist() + [1]
        slist = [self.sigma] * (self.kernel_num-1) + [self.exact_sigma]

        return torch.nn.ModuleList([GaussianKernel(m, s) for m, s in zip(mlist, slist)])

    def _get_mlp(self) -> torch.nn.Sequential:
        """
        Формирует выходной MLP-слой для ранжирования на основе результата Kernels.
        Точная структура зависит от аттрибута out_layers.
        - Если out_layers = [], то MLP становится
        линейным слоем из K (признаки равны результатам ядер) в 1 (финальный скор релевантности).
        - Если out_layers = [10, 5], то архитектура следующая: K->ReLU->10->ReLU->5->ReLU->1.
        ps: нелинейность не применяется в конце MLP. Таким образом, с помощью цикла нужно
        научиться в автоматическом режиме генерировать архитектуру выходного слоя.
        """
        if not self.out_layers:
            output = [torch.nn.Linear(self.kernel_num, 1)]
        else:
            layer_sizes = self.out_layers + [1]
            output = [torch.nn.Linear(self.kernel_num, layer_sizes[0])]
            for current, previous in zip(layer_sizes[:-1], layer_sizes[1:]):
                output += [torch.nn.ReLU(), torch.nn.Linear(current, previous)]

        return torch.nn.Sequential(*output)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        """
        Формирует матрицу взаимодействия “каждый-с-каждым” между словами одного и второго
        вопроса (запрос и документ). В качестве меры используется косинусная схожесть
        (cosine similarity) между эмбеддингами отдельных токенов.
        """
        batch_size = query.shape[0]
        cosines = []
        
        query = self.embeddings(query)
        doc = self.embeddings(doc)

        cosines = []

        for i in range(batch_size):
            for qi in range(query.shape[1]):
                cosine = self.cos(query[i, qi], doc[i])
                cosines.append(cosine)
        # [Batch, Left, Right]
        cosines = torch.vstack(cosines).reshape(batch_size, query.shape[1], doc.shape[1])
        return cosines

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        """
        Применяет ядра к matching_matrix. Метод реализован.
        """
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left, Embedding], [Batch, Right, Embedding]
        query, doc = inputs['query'], inputs['document']
        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out

########
# Блок 4
########


class RankingDataset(torch.utils.data.Dataset):
    """
    Отвечает за обработку текстов.

    три уровня релевантности:
    2 - дубликат (согласно оригинальной разметке это пары с таргетом, равным 1)
    1 - отдаленно похож (согласно оригинальной разметке это пары с таргетом, равным 0)
    0 - нерелевантный (таких пар в датасете нет, их можно нагенерировать самостоятельно)
    """

    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        """
        index_pairs_or_triplets - список списков id (и, конечно, лейблов).

        idx_to_text_mapping - соотнесение индекса (id_left и id_right) с текстом,
          подробнее в методе get_idx_to_text_mapping класса Solution.

        vocab - маппинг слова в индекс которые подаются в эмбеддинг-слой KNRM.

        oov_val - значение (индекс)  в словаре на случай, если слово не представлено в словаре.

        preproc_func - функция обработки и токенизации текста.

        max_len - максимальное количество токенов в тексте.
        """
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        """
        Перевод обработанного текста после preproc_func в индексы.
        """
        return [self.vocab.get(t, 1) for t in tokenized_text[:self.max_len]]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        """
        Перевод id_left/id_right в индексы токенов в словаре с помощью функции _tokenized_text_to_index.
        """
        tokenized_text = self.preproc_func(self.idx_to_text_mapping[idx])
        return self._tokenized_text_to_index(tokenized_text)

    def __getitem__(self, idx: int):
        """
        Возвращает набор признаков для заданной пары или триплета.
        Несколько id и таргет, где признаки выражены индексами слов в словаре.
        """
        pass


class TrainTripletsDataset(RankingDataset):
    """
    (исходный вопрос, вопрос-кандидат1 и  вопрос-кандидат2 для обучения в PairWise-режиме)
    """

    def __getitem__(self, idx):
        """
        Два словаря с ключами query и document, а также целевая метка - ответ на вопрос
        "правда ли, что первый документ более релевантен запросу, чем второй?"
        """
        q, doc_l, doc_r, label = self.index_pairs_or_triplets[idx]
        q = self._convert_text_idx_to_token_idxs(q)
        doc_l = self._convert_text_idx_to_token_idxs(doc_l)
        doc_r = self._convert_text_idx_to_token_idxs(doc_r)
        l = {'query': q, 'document': doc_l}
        r = {'query': q, 'document': doc_r}
        return l, r, label


class ValPairsDataset(RankingDataset):
    """
    (оценивает отдельно релевантность вопроса-кандидата к исходному вопросу)
    Метод для генерации пар create_val_pairs класса Solution
    """

    def __getitem__(self, idx):
        """
        Словарь с ключами query и document, а также целевая метка - релевантность от 0 до 2.
        """
        doc_l, doc_r, label = self.index_pairs_or_triplets[idx]
        doc_l = self._convert_text_idx_to_token_idxs(doc_l)
        doc_r = self._convert_text_idx_to_token_idxs(doc_r)
        return {'query': doc_l, 'document': doc_r}, label


def collate_fn(batch_objs: List[Union[Dict[str, torch.Tensor], torch.FloatTensor]]):
    """
    Собирать батч из нескольких тренировочных примеров для KNRM.

    batch_objs - список из выходов датастов и формирует из них единый dict с тензорами в качестве значений.
    """
    max_len_q1 = -1
    max_len_d1 = -1
    max_len_q2 = -1
    max_len_d2 = -1

    is_triplets = False
    for elem in batch_objs:
        if len(elem) == 3:
            left_elem, right_elem, label = elem
            is_triplets = True
        else:
            left_elem, label = elem

        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)
        if len(elem) == 3:
            max_len_q2 = max(len(right_elem['query']), max_len_q2)
            max_len_d2 = max(len(right_elem['document']), max_len_d2)

    q1s = []
    d1s = []
    q2s = []
    d2s = []
    labels = []

    for elem in batch_objs:
        if is_triplets:
            left_elem, right_elem, label = elem
        else:
            left_elem, label = elem

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])
        if is_triplets:
            pad_len3 = max_len_q2 - len(right_elem['query'])
            pad_len4 = max_len_d2 - len(right_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
        if is_triplets:
            q2s.append(right_elem['query'] + [0] * pad_len3)
            d2s.append(right_elem['document'] + [0] * pad_len4)
        labels.append([label])
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)
    if is_triplets:
        q2s = torch.LongTensor(q2s)
        d2s = torch.LongTensor(d2s)
    labels = torch.FloatTensor(labels)

    ret_left = {'query': q1s, 'document': d1s}
    if is_triplets:
        ret_right = {'query': q2s, 'document': d2s}
        return ret_left, ret_right, labels
    else:
        return ret_left, labels

############
# Блок 1,2,5
############


class Solution:
    def __init__(self, glue_qqp_dir: str, glove_vectors_path: str,
                 min_token_occurancies: int = 1,
                 random_seed: int = 0,
                 emb_rand_uni_bound: float = 0.2,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 dataloader_bs: int = 1024,
                 train_lr: float = 0.001,
                 change_train_loader_ep: int = 10
                 ):
        """
        min_token_occurancies - минимальное количество раз, которое слово (токен) должно появиться
          в выборке, чтобы не быть отброшенным как низкочастотное.

        emb_rand_uni_bound - половина ширины интервала, из которого равномерно (uniform)
          генерируются вектора эмбеддингов (если вектор не представлен в наборе GloVe).
          Если параметр равен 0.2, то каждая компонента вектора принадлежит U(−0.2,0.2)

        freeze_knrm_embeddings - флаг, указывающий на необходимость дообучения эмбеддингов,
          будут ли по ним считаться градиенты (при True дообучение происходить не будет)

        knrm_out_mlp - конфигурация MLP-слоя на выходе в KNRM.

        dataloader_bs - размер батча при Обучении и Валидации модели.

        change_train_loader_ep - частота "менять/перегенерировать" выборку для трнировки модели.
        """
        self.trans = str.maketrans(string.punctuation, " " * len(string.punctuation))

        self.glue_qqp_dir = glue_qqp_dir
        self.glove_vectors_path = glove_vectors_path
        self.glue_train_df = self.get_glue_df('train')
        self.glue_dev_df = self.get_glue_df('dev')
        self.dev_pairs_for_ndcg = self.create_val_pairs(self.glue_dev_df)
        self.min_token_occurancies = min_token_occurancies
        self.all_tokens = self.get_all_tokens([self.glue_train_df, self.glue_dev_df], self.min_token_occurancies)

        self.random_seed = random_seed
        self.emb_rand_uni_bound = emb_rand_uni_bound
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.dataloader_bs = dataloader_bs
        self.train_lr = train_lr
        self.change_train_loader_ep = change_train_loader_ep

        self.model, self.vocab, self.unk_words = self.build_knrm_model()

        self.idx_to_text_mapping_train = self.get_idx_to_text_mapping(self.glue_train_df)
        self.idx_to_text_mapping_dev = self.get_idx_to_text_mapping(self.glue_dev_df)

        self.val_dataset = ValPairsDataset(self.dev_pairs_for_ndcg,
                                           self.idx_to_text_mapping_dev,
                                           vocab=self.vocab, oov_val=self.vocab['OOV'],
                                           preproc_func=self.simple_preproc)
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.dataloader_bs, num_workers=0,
            collate_fn=collate_fn, shuffle=False)

    def get_glue_df(self, partition_type: str) -> pd.DataFrame:
        assert partition_type in ['dev', 'train']
        glue_df = pd.read_csv(
            self.glue_qqp_dir + f'/{partition_type}.tsv', sep='\t', error_bad_lines=False, dtype=object)
        glue_df = glue_df.dropna(axis=0, how='any').reset_index(drop=True)
        glue_df_fin = pd.DataFrame({
            'id_left': glue_df['qid1'].astype('int32'),
            'id_right': glue_df['qid2'].astype('int32'),
            'text_left': glue_df['question1'],
            'text_right': glue_df['question2'],
            'label': glue_df['is_duplicate'].astype('int32')
        })
        return glue_df_fin

    ########
    # Блок 1
    ########

    def hadle_punctuation(self, inp_str: str) -> str:
        """
        Очищает строку от пунктуации. Все знаки пунктуации необходимо взять из string.punctuation.
        Подумайте, какая именно необходима замена знакам пунктуации.
        """
        return inp_str.translate(self.trans)

    def simple_preproc(self, inp_str: str) -> List[str]:
        """
        Полный препроцессинг строки. Должно включать в себя обработку пунктуации и приведение
        к нижнему регистру, а в качестве токенизации используется nltk.word_tokenize.

        Returns: лист со строками (токенами)
        """
        return nltk.word_tokenize(self.hadle_punctuation(inp_str).lower())

    def _filter_rare_words(self, vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
        """
        Отсечь те, которые не проходят порог, равный min_token_occurancies
        """
        return [k for k, v in vocab.items() if v >= min_occurancies]

    def gen_unique_text(self, dflist):
        from tqdm.auto import tqdm
        idset = set()
        for df in dflist:
            dfcut = df[~df.id_left.isin(idset) | ~df.id_right.isin(idset)]
            for _, row in tqdm(dfcut.iterrows(), total=dfcut.shape[0]):
                if row.id_left not in idset:
                    yield row.text_left
                    idset.add(row.id_left)
                if row.id_right not in idset:
                    yield row.text_right
                    idset.add(row.id_right)

    def get_all_tokens(self, list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
        """
        Метод формирующий список ВСЕХ токенов, представленных в подаваемых на вход датасетах.
        Необходимо сформировать уникальное множество всех текстов, затем рассчитать частотность
        каждого токена (то есть после обработки simple_preproc) и отсечь те, которые не проходят
        порог, равный min_token_occurancies.

        Returns: список токенов, для которых будут формироваться эмбеддинги и на которые будут
        разбиваться оригинальные тексты вопросов.
        """
        fpath = "/home/nur/projects/analysis/range_matching/data/token_list.txt"
        import os  # ToDo delete
        if os.path.exists(fpath):
            with open(fpath, 'r') as f:
                return [row.strip() for row in f if row]

        gen_token = (t for text in self.gen_unique_text(list_of_df) for t in self.simple_preproc(text))
        counter = Counter(gen_token)
        res = self._filter_rare_words(counter, min_occurancies)

        # with open(fpath, 'w') as f:
        #     for t in res:
        #         f.write(t+'\n')
        return res

    #######
    # Блок2
    #######

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        """
        Считывание файла эмбеддингов в словарь
        """
        with open(file_path, 'r') as f:
            return dict((row.strip().split(' ', 1) for row in f))

    def create_glove_emb_from_file(self, file_path: str, inner_keys: List[str],
                                   random_seed: int, rand_uni_bound: float
                                   ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """
        Метод формирует (1) матрицу эмбеддингов размера N∗D, (2) словарь размера N,
        сопоставляющий каждому слову индекс эмбеддинга, (3) список слов, которые не были в
        исходных эмбеддингах (генерировать случайный эмбеддинг из равномерного распределения
        или другой вектор с заданными характеристиками).
        ps: необходимо в словарь добавить два специальных токена - PAD и OOV, с индексами 0 и 1.
        'PAD' используется для заполнения пустот в тензорах (когда один вопрос состоит из
        бОльшего количества токенов, чем второй, однако их необходимо представить в виде матрицы,
        в которой строки имеют одинаковую длину) и должен состоять полностью из нулей.
        'OOV' для токенов, которых нет в словаре.

        Returns: На выходе в матрице эмбеддингов (и в словаре) должны быть как загруженные из
        файла вектора (для тех слов, которые в нем встретились), так и для новых слов (из unk_words,
        включая PAD и OOV).
        ex: для min_token_occurancies=1 доля unk_words из всех слов должна быть около 30%.
        """
        d = self._read_glove_embeddings(file_path)
        embs = [np.zeros(50, dtype=np.float32), np.random.uniform(size=50)]
        vocab = {'PAD': 0, 'OOV': 1}
        unk_words = ['PAD', 'OOV']
        for i, t in enumerate(inner_keys, start=2):
            if t not in d:
                unk_words.append(t)
                emb = np.random.uniform(size=50)
            else:
                emb = np.array(list(map(float, d[t].split())), dtype=np.float32)
            embs.append(emb)
            vocab[t] = i
        matrix = np.asarray(embs, dtype=np.float32)
        return matrix, vocab, unk_words

    ########
    # Блок 4
    ########

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        emb_matrix, vocab, unk_words = self.create_glove_emb_from_file(
            self.glove_vectors_path, self.all_tokens, self.random_seed, self.emb_rand_uni_bound)
        torch.manual_seed(self.random_seed)
        knrm = KNRM(emb_matrix, freeze_embeddings=self.freeze_knrm_embeddings,
                    out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        return knrm, vocab, unk_words

    def create_val_pairs(self, inp_df: pd.DataFrame, fill_top_to: int = 15,
                         min_group_size: int = 2, seed: int = 0) -> List[List[Union[str, float]]]:
        inp_df_select = inp_df[['id_left', 'id_right', 'label']]
        inf_df_group_sizes = inp_df_select.groupby('id_left').size()
        glue_dev_leftids_to_use = list(
            inf_df_group_sizes[inf_df_group_sizes >= min_group_size].index)
        groups = inp_df_select[inp_df_select.id_left.isin(
            glue_dev_leftids_to_use)].groupby('id_left')

        all_ids = set(inp_df['id_left']).union(set(inp_df['id_right']))

        out_pairs = []

        np.random.seed(seed)

        for id_left, group in groups:
            ones_ids = group[group.label > 0].id_right.values
            zeroes_ids = group[group.label == 0].id_right.values
            sum_len = len(ones_ids) + len(zeroes_ids)
            num_pad_items = max(0, fill_top_to - sum_len)
            if num_pad_items > 0:
                cur_chosen = set(ones_ids).union(
                    set(zeroes_ids)).union({id_left})
                pad_sample = np.random.choice(
                    list(all_ids - cur_chosen), num_pad_items, replace=False).tolist()
            else:
                pad_sample = []
            for i in ones_ids:
                out_pairs.append([id_left, i, 2])
            for i in zeroes_ids:
                out_pairs.append([id_left, i, 1])
            for i in pad_sample:
                out_pairs.append([id_left, i, 0])
        return out_pairs

    def get_idx_to_text_mapping(self, inp_df: pd.DataFrame) -> Dict[str, str]:
        left_dict = (
            inp_df
            [['id_left', 'text_left']]
            .drop_duplicates()
            .set_index('id_left')
            ['text_left']
            .to_dict()
        )
        right_dict = (
            inp_df
            [['id_right', 'text_right']]
            .drop_duplicates()
            .set_index('id_right')
            ['text_right']
            .to_dict()
        )
        left_dict.update(right_dict)
        return left_dict

    ########
    # Блок 5
    ########

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

    def _ndcg_k(self,  ys_true: torch.Tensor, ys_pred: torch.Tensor, ndcg_top_k=10) -> float:
        ys_true = torch.FloatTensor(ys_true)
        ys_pred = torch.FloatTensor(ys_pred)
        current_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        if ideal_dcg > 0:
            return current_dcg / ideal_dcg
        else:
            return 0

    def valid(self, model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader) -> float:
        labels_and_groups = val_dataloader.dataset.index_pairs_or_triplets
        labels_and_groups = pd.DataFrame(labels_and_groups, columns=['left_id', 'right_id', 'rel'])

        all_preds = []
        for batch in (val_dataloader):
            inp_1, y = batch
            preds = model.predict(inp_1)
            preds_np = preds.detach().numpy()
            all_preds.append(preds_np)
        all_preds = np.concatenate(all_preds, axis=0)
        labels_and_groups['preds'] = all_preds

        ndcgs = []
        for cur_id in labels_and_groups.left_id.unique():
            cur_df = labels_and_groups[labels_and_groups.left_id == cur_id]
            ndcg = self._ndcg_k(cur_df.rel.values.reshape(-1), cur_df.preds.values.reshape(-1))
            if np.isnan(ndcg):
                ndcgs.append(0)
            else:
                ndcgs.append(ndcg)
        return np.mean(ndcgs)

    def sample_data_for_train_iter(self, inp_df: pd.DataFrame, seed: int) -> List[List[Union[str, float]]]:
        """
        Нужно определять, релевантнее ли второй документ, чем первый, к данному запросу.
        Returns: тренировочная выборка.
        """
        import random
        qd_rel = self.create_val_pairs(inp_df, seed=seed)
        df = pd.DataFrame.from_records(qd_rel)
        df.columns = ['q', 'd', 'rel']
        random.seed(seed)
        triplets_pos = list()
        triplets_neg = list()
        for q in df.q.unique():
            pos = df[(df.q == q) & (df.rel > 0)].d.unique()
            neg = df[(df.q == q) & (df.rel == 0)].d.unique()
            for p in pos:
                for n in neg:
                    triplets_pos.append([q, p, n, 1.])
                    triplets_neg.append([q, n, p, 0.])
            pos_1 = df[(df.q == q) & (df.rel == 1)].d.unique()
            pos_2 = df[(df.q == q) & (df.rel == 2)].d.unique()
            for p in pos_2:
                for n in pos_1:
                    triplets_pos.append([q, p, n, 1.])
                    triplets_neg.append([q, n, p, 0.])

        sample_count = self.change_train_loader_ep * self.dataloader_bs // 2
        triples = random.sample(triplets_pos, sample_count) + random.sample(triplets_neg, sample_count)
        random.shuffle(triples)
        return triples

    def train(self, n_epochs: int):
        """
        Обучаться в PairWise-режиме

        N итераций по тренировочному Dataloader. В зависимости от метода формирования тренировочной
        выборки, по необходимости пересоздавайте выборку каждые change_train_loader_ep эпох.
        Реализовать методику подбора триплетов документов для обучения в PairWise-режиме нейросети KNRM.
        ex: создание валидационного пулла в качестве примера.
        ex2: генерировать порядка 8-10 тысяч триплетов для обучения.
        Это недетерминированынй процесс и потому появляется возможность каждые 'K' эпох менять выборку
        (не меняя смысла задачи - всё еще нужно определять, релевантнее ли второй документ, чем первый,
        к данному запросу), например - менять left и right id.
        """
        from tqdm.auto import tqdm
        opt = torch.optim.SGD(self.model.parameters(), lr=self.train_lr)
        criterion = torch.nn.BCELoss()

        # with torch.no_grad():
        #     val_metric = self.valid(self.model, self.val_dataloader)
        #     print("Val metric (0) =", val_metric)

        self.model.train()
        for epoch in tqdm(range(n_epochs), total=n_epochs):
            if epoch % self.change_train_loader_ep == 0:
                start_i = epoch // self.change_train_loader_ep
                count = 20000
                train_triplets = self.sample_data_for_train_iter(self.glue_train_df.iloc[start_i*count: (start_i+1) *count], seed=epoch)

                train_dataset = TrainTripletsDataset(train_triplets,
                                                     self.idx_to_text_mapping_train,
                                                     vocab=self.vocab, oov_val=self.vocab['OOV'],
                                                     preproc_func=self.simple_preproc)
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.dataloader_bs, num_workers=0,
                    collate_fn=collate_fn, shuffle=False)

            for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                opt.zero_grad()

                d1, d2, batch_target = batch
                batch_pred = self.model(d1, d2)
                batch_loss = criterion(batch_pred, batch_target)
                batch_loss.backward(retain_graph=True)
                opt.step()

            with torch.no_grad():
                self.model.eval()
                val_metric = self.valid(self.model, self.val_dataloader)
                print(f"Val metric ({epoch}) = {val_metric}")


if __name__ == "__main__":
    """
    Преодолеть порог в 0.925. Правильное решение даже без смены набора триплетов
    8-12 итерация по датасету размера 8-10 тысяч с batch_size 1024.
    Модель при отправке тренируется с нуля до 20 эпох не более 7 минут.
    """
    s = Solution(glue_qqp_dir, glove_path)
    s.train(n_epochs=20)
    print(f"Trained model NDCG={metric}")
