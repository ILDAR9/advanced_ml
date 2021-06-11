# -*- coding: utf-8 -*-
"""
Система подсказок похожих вопросов на уже знакомых вам данных сайта Quora
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
import pickle
import string
from collections import Counter
from typing import Callable, Dict, List, Tuple, Union, Optional
from flask import Flask
from dotenv import load_dotenv, find_dotenv
import pickle
load_dotenv(find_dotenv())  # ToDo delete


class FAQ:
    def __init__(self, num_candidates=15):
        """
        120 секунд на инициализацию
        emb_path_knrm is object of 'Solution.().model.embeddings.state_dict()'
        Порядок токенов соответсвует VOCAB_PATH: ключ - слово, значение - индекс в матрице эмбеддингов.


        """
        with open(os.environ['VOCAB_PATH'], 'r') as f:
            self.vocab = json.load(f)
        self._init_knrm(os.environ['EMB_PATH_KNRM'], os.environ['MLP_PATH'])
        self._read_knrm_embeddings()
        self.glove: Dict[str, np.array] = self._read_glove_embeddings(os.environ['EMB_PATH_GLOVE'])
        self._is_ready = False
        self.num_candidates = num_candidates

    def prepare(self):  # ToDo delete
        import time
        time.sleep(5)
        self._is_ready = True

    @staticmethod
    def is_english(input_str: str) -> bool:
        return detect(input_str) == 'en'

    def is_ready(self):
        return self._is_ready

    def init_index(self, doc_dict: Dict[str, str]):
        """
        use Glove
        """
        doc_dict  # preprocess and build embeddings

        dim = emb_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb_matrix)
        return self.index.ntotal

    def get_candidates(self, vec: np.ndarray):
        D, I = self.index.search(vec, self.num_candidates)  # Indices, Distances
        return I, D

    def hadle_punctuation(self, inp_str: str) -> str:
        return inp_str.translate(self.trans)

    def simple_preproc(self, inp_str: str) -> List[str]:
        return nltk.word_tokenize(self.hadle_punctuation(inp_str).lower())

    def _init_knrm(self, knrm_emb_path: str, mlp_path: str):
        with open(knrm_emb_path, 'r') as f:
            emb_state = pickle.load(f)
        # ToDo finish load emb state dict Solution.().model.embeddings.state_dict()'
        mlp_path

    def _read_glove_embeddings(self, file_path: str) -> Dict[str, np.array]:
        """
        Считывание файла эмбеддингов в словарь
        """
        emb_d = dict()
        with open(file_path, 'r') as f:
            k, emb_str = (row.strip().split(' ', 1) for row in f)
            emb = np.array(list(map(float, emb_str.split())), dtype=np.float32)
            emb_d[k] = emb
        return emb_d

    def get_topn(query: str) -> List[Tuple[str, str]]:
        """
        Поиск вопросов-кандидатов с помощью FAISS (по схожести векторов) - в этой части
        предлагается ограничиться векторизацией только тех слов, чьи эмбеддинги есть в исходных
        GLOVE-векторах. Эти кандидаты реранжируются KNRM-моделью, после чего до 10 кандидатов
        выдаются в качестве ответа.

        Returns: до 10 найденных схожих вопросов, где каждый вопрос представлен в виде Tuple,
        в котором первое значение - id текста, второе - сам НеПредобработанный текст схожего вопроса.
        """
        pass

    def is_candidates_initialized():
        return self.index != None


app = Flask(__name__)
faq = FAQ()


@app.route("/ping", methods=['GET'])
def ping():
    if faq.is_ready():
        return {"status": "ok"}


@app.route('/update_index', methods=['POST'])
def update_index(docdict):
    """
    ключ - id текста, значение - сам текст
    200 секунд для предобработки и создание индекса
    """
    docs: Dict[str, str] = json.loads(request.json)['documents']
    total = faq.init_index(docs)
    return {"status": "ok", "index_size": total}


@app.route('/query', methods=['POST'])
def query(request):
    if not faq.is_candidates_initialized():
        return {'status': 'FAISS is not initialized!'}
    queries: Dict[str, List[str]] = json.loads(request.json)['queries']
    is_eng_list: List[bool] = list(map(faq.is_english, queries))
    suggestions: List[Optional[List[Tuple[str, str]]]] = []
    for query, is_eng in is_eng_list:
        if not is_eng:
            suggestions.append(None)
        else:
            topn_suggest = faq.get_topn(query)
            suggestions.append(topn_suggest)

    return {"lang_check": is_eng_list, "suggestions": suggestions}


if __name__ == "__main__":
    # vec = np.random.rand(1, 50).astype('float32')
    # print(faq.get_candidates(vec))
    # input_str = "Hi my dear friend what is this"
    # print(faq.is_english(input_str))

    # FLASK_APP=faq_service.py flask run --port 11000
