'''
Модуль со всем необходимым для подключения и работы с БД.
'''

import dask.dataframe as dd
from typing import Dict


class Engine:
    def __init__(self, tables: Dict[str, dd.DataFrame]):
        self.tables = tables

    def register_table(self, table: dd.DataFrame, name: str) -> None:
        self.tables[name] = table

    def get_table(self, name: str) -> dd.DataFrame:
        return self.tables[name]
