import pickle
import json


def load_pickle(path: str):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def dump_pickle(obj, path: str):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def load_json(path: str):
    with open(path, 'r') as file:
        obj = json.load(file)
    return obj
