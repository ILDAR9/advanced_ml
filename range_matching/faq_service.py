# -*- coding: utf-8 -*-
"""
Система подсказок похожих вопросов на данных сайта Quora
"""
from langdetect import detect
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import nltk
import faiss
import json
import nltk
import os
import math
from collections import defaultdict
import string
from collections import Counter
from typing import Callable, Dict, List, Tuple, Union, Optional
from flask import Flask
from flask import request

# ToDo delete
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())


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
        return torch.exp(-0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2))


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
        step = 1. / (self.kernel_num-1)
        mlist = np.arange(-1, 1, step)[1::2].tolist() + [1]
        slist = [self.sigma] * (self.kernel_num-1) + [self.exact_sigma]

        return torch.nn.ModuleList([GaussianKernel(m, s) for m, s in zip(mlist, slist)])

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = []
        for in_f, out_f in zip(out_cont, out_cont[1:]):
            mlp += [torch.nn.Linear(in_f, out_f), torch.nn.ReLU()]
        return torch.nn.Sequential(*mlp[:-1])

        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

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


def collate_fn(batch_objs: List[Dict[str, torch.Tensor]]):
    """
    Собирать батч из нескольких тренировочных примеров для KNRM.

    batch_objs - список из выходов датастов и формирует из них единый dict с тензорами в качестве значений.
    """
    max_len_q1 = -1
    max_len_d1 = -1

    is_triplets = False
    for left_elem in batch_objs:
        max_len_q1 = max(len(left_elem['query']), max_len_q1)
        max_len_d1 = max(len(left_elem['document']), max_len_d1)

    q1s = []
    d1s = []

    for left_elem in batch_objs:

        pad_len1 = max_len_q1 - len(left_elem['query'])
        pad_len2 = max_len_d1 - len(left_elem['document'])

        q1s.append(left_elem['query'] + [0] * pad_len1)
        d1s.append(left_elem['document'] + [0] * pad_len2)
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)

    ret_left = {'query': q1s, 'document': d1s}
    return ret_left


class FAQ:
    def __init__(self, num_candidates=20):
        """
        120 секунд на инициализацию
        """
        self.num_candidates = num_candidates
        self.knrm = self._init_knrm()
        self.index = None
        self.idx_to_text_mapping = None

        self.trans = str.maketrans(string.punctuation, " " * len(string.punctuation))
        self.glove_embs, self.dim = self._read_glove_embeddings(os.environ['EMB_PATH_GLOVE'])
        with open(os.environ['VOCAB_PATH'], 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

    @staticmethod
    def _init_knrm():
        with open(os.environ['EMB_PATH_KNRM'], 'rb') as f:
            emb_state_dict = torch.load(f)
        model = KNRM(emb_state_dict['weight'], freeze_embeddings=True, out_layers=[], kernel_num=21)
        with open(os.environ['MLP_PATH'], 'rb') as f:
            model.mlp.load_state_dict(torch.load(f))
        return model

    @staticmethod
    def is_english(input_str: str) -> bool:
        return detect(input_str) == 'en'

    def is_candidates_initialized(self) -> bool:
        return self.index is not None

    def hadle_punctuation(self, inp_str: str) -> str:
        return inp_str.translate(self.trans)

    def simple_preproc(self, inp_str: str) -> List[str]:
        return nltk.word_tokenize(self.hadle_punctuation(inp_str).lower())

    ################
    # FAQ candidates
    ################

    def _sentence_emb(self, text: str) -> np.array:
        tokens = self.simple_preproc(text)
        emb_list = [self.glove_embs[t] for t in tokens if t in self.glove_embs]
        if emb_list:
            emb_avg = np.average(emb_list, axis=0)
        else:
            emb_avg = np.zeros(self.dim)
        return emb_avg.astype('float32').reshape(1, -1)

    def update_index(self, doc_dict: Dict[str, str]):
        self.idx_to_text_mapping = {int(k): v for k, v in doc_dict.items()}
        self.index = None
        idx_list = []
        emb_list = []
        for idx, text in self.idx_to_text_mapping.items():
            text_emb = self._sentence_emb(text)
            idx_list.append(idx)
            emb_list.append(text_emb)

        idxs = np.vstack(idx_list).reshape(-1)
        embs = np.vstack(emb_list)

        index = faiss.IndexFlatL2(self.dim)
        # index = faiss.index_factory(self.dim, "IVF500,Flat")
        index = faiss.IndexIDMap(index)
        # training_vectors = embs
        # index.train(training_vectors)
        index.add_with_ids(embs, idxs)
        self.index = index

        return self.index.ntotal

    def get_candidates(self, query: str) -> List[int]:
        emb = self._sentence_emb(query)
        _, I = self.index.search(emb, self.num_candidates)  # Distances, Indices
        return np.array([x for x in I.squeeze() if x > -1])

    def _read_glove_embeddings(self, file_path: str) -> Tuple[Dict[str, np.array], int]:
        """
        Считывание файла эмбеддингов в словарь
        """
        emb_d = dict()
        with open(file_path, 'r') as f:
            for row in f:
                k, emb_str = row.strip().split(' ', 1)
                emb = np.array(list(map(float, emb_str.split())), dtype=np.float32)
                emb_d[k] = emb
        dim = next(iter(emb_d.values())).shape[-1]
        return emb_d, dim

    ######
    # KNRM
    ######

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        return [self.vocab.get(t, 1) for t in tokenized_text]

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        tokenized_text = self.simple_preproc(self.idx_to_text_mapping[idx])
        return self._tokenized_text_to_index(tokenized_text)

    # def rerank(self, candidates: List[int]):
    #     counter = defaultdict(lambda: 0)
    #     query_list = []
    #     doc_list = []
    #     for i, a in enumerate(candidates):
    #         for b in candidates[i:]:
    #             doc_l = self._convert_text_idx_to_token_idxs(a)
    #             doc_r = self._convert_text_idx_to_token_idxs(b)
    #             doc_l = torch.reshape(torch.LongTensor(doc_l), (1, -1))
    #             doc_r = torch.reshape(torch.LongTensor(doc_r), (1, -1))

    #             inp = {'query': doc_l, 'document': doc_r}
    #             r = self.knrm.predict(inp).detach().numpy()
    #             if r > 0:
    #                 counter[a] += 1
    #                 counter[b] -= 1
    #             elif r < 0:
    #                 counter[a] -= 1
    #                 counter[b] += 1

    #     return sorted(counter.keys(), key=counter.get, reverse=True)

    def rerank(self, query:str, candidates: np.array):
        counter = defaultdict(lambda: 0)
        inp_list = []

        query_tok_idx = self._tokenized_text_to_index(self.simple_preproc(query))
        for cand in candidates:
            inp = {'query': query_tok_idx,
                    'document': self._convert_text_idx_to_token_idxs(cand)}
            inp_list.append(inp)

        inp = collate_fn(inp_list)
        preds = self.knrm.predict(inp).squeeze()
        _, idxs = torch.sort(preds, descending=True)
        return candidates[idxs]
        

    ############
    # Controller
    ############

    def query(self, queries: Dict[str, List[str]], N=10):
        """
        Returns: до 10 найденных схожих вопросов, где каждый вопрос представлен в виде Tuple,
        в котором первое значение - id текста, второе - сам НеПредобработанный текст схожего вопроса.
        """
        is_eng_list: List[bool] = list(map(self.is_english, queries))
        suggestions: List[Optional[List[Tuple[str, str]]]] = []
        for query, is_eng in zip(queries, is_eng_list):
            if not is_eng:
                suggestions.append(None)
            else:
                candidates = self.get_candidates(query)
                ranked_cands = self.rerank(query, candidates)[:N]
                suggest = [(str(idx), self.idx_to_text_mapping[idx]) for idx in ranked_cands]
                suggestions.append(suggest)

        return is_eng_list, suggestions


app = Flask(__name__)
faq = None


@app.route("/ping", methods=['GET'])
def ping():
    res = {"status": "ok"}
    return json.dumps(res)


@app.route('/update_index', methods=['POST'])
def update_index():
    """
    ключ - id текста, значение - сам текст
    200 секунд для предобработки и создание индекса
    """
    # docs: Dict[str, str] = request.json['documents']
    docs: Dict[str, str] = json.loads(request.json)['documents']
    global faq
    if faq is None:
        faq = FAQ(num_candidates=20)
    total = faq.update_index(docs)
    res = {"status": "ok", "index_size": total}
    return json.dumps(res)


@app.route('/query', methods=['POST'])
def query():
    global faq
    if faq is None or not faq.is_candidates_initialized():
        res = {'status': 'FAISS is not initialized!'}
        return json.dumps(res)
    queries: Dict[str, List[str]] = json.loads(request.json)['queries']
    # queries: Dict[str, List[str]] = request.json['queries']
    is_eng_list, suggestions = faq.query(queries)
    res = {"lang_check": is_eng_list, "suggestions": suggestions}
    return json.dumps(res)


"""
FLASK_APP=faq_service.py flask run --port 11000
"""
if __name__ == "__main__":
    app.run()
    # faq = FAQ()
    # fpath = "/home/nur/projects/analysis/range_matching/data/documents.json"
    # with open(fpath, 'r', encoding='utf-8') as f:
    #     docs: Dict[str, str] = json.load(f)['documents']

    # assert not faq.is_candidates_initialized()
    # total = faq.update_index(docs)
    # print('total', total)
    # assert faq.is_candidates_initialized()

    # queries = ['How long to do phd?', 'Where is my photo?']
    # is_eng_list, suggestions = faq.query(queries)
    # print(is_eng_list)
    # print(suggestions)
