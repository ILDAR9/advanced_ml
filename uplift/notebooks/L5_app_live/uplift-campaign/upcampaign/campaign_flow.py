import os
import sys
import pandas as pd
import dask.dataframe as dd
import datetime
import sklearn.pipeline as skpipe
import logging

from typing import List, Dict, Optional

import upcampaign.model_utils as mu
import upcampaign.datalib.features.compute as fcompute
import upcampaign.datalib.features.base as fbase

from .datalib.connection import Engine
from .utils.flow import Flow
from .custom_transforms import ModelApplyTransformer
from .utils.data import load_pickle, dump_pickle, load_json
from .utils.logging import SILENT_LOGGER


N_DAYS_CHURN = 30


def select_suitable_clients(engine: Engine, n_clients: int, date_to: datetime.date) -> dd.DataFrame:
    receipts = engine.get_table('receipts')

    date_to = datetime.datetime.combine(date_to, datetime.datetime.min.time())
    date_from = date_to - datetime.timedelta(days=N_DAYS_CHURN)
    date_mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

    clients = (
        receipts
        .loc[date_mask]['client_id']
        .unique().compute()
        .sample(n=n_clients)
        .reset_index(drop=True).to_frame()
    )
    return clients


def create_transform_pipeline(config: list, artifacts_root_path: str) -> skpipe.Pipeline:
    steps = list()
    for i, part_config in enumerate(config):
        if part_config['type'] == 'pipeline_pickle':
            part = load_pickle(os.path.join(artifacts_root_path, part_config['path']))
            steps.extend(part.steps)
        elif part_config['type'] == 'pipeline_json':
            part = fcompute.create_pipeline(
                transform_config=load_json(os.path.join(artifacts_root_path, part_config['path'])),
                transformer_reference=fcompute.TRANSFORMER_REFERENCE
            )
            steps.extend(part.steps)
        elif part_config['type']  == 'model_apply':
            transformer = ModelApplyTransformer(
                model=mu.load_model(os.path.join(artifacts_root_path, part_config['model_path'])),
                col_result=part_config['col_result']
            )
            steps.append((part_config.get('name', f'part_{i}'), transformer))
        else:
            transformer = fcompute.create_transformer(
                part_config['type'],
                transformer_reference=fcompute.TRANSFORMER_REFERENCE,
                **part_config['args']
            )
            steps.append((part_config.get('name', f'part_{i}'), transformer))
    pipeline = skpipe.Pipeline(steps)
    return pipeline


def split_into_groups(data: pd.DataFrame, group_sizes: List[int], random_state=None) -> List[pd.DataFrame]:
    data_ = data.sample(frac=1, random_state=random_state)
    result = list()
    offset = 0
    for size in group_sizes:
        result.append(data_.iloc[offset: offset + size])
        offset += size
    return result


class CampaignFlow(Flow):
    """
    Класс запуска расчетов, необходимых для старта кампании. 
    
    * Параметризуется "большим" конфигом config.
    * Расчет состоит из нескольких этапов.
    * Есть возможность перезапуска расчета с последнего успешного этапа.
    * Сохраняет промежуточные результаты на жесткий диск.
    
    Args:
        engine: соединение с БД.
        config: конфигурация расчета.
        filenames: словарь названий файлов для промежуточных результатов.
        run_id: id расчета. По нему восстанавливаются промежуточные результаты при переподнятии процесса.
        runs_root_path: папка, в которой создастся подпапка для расчета.
        artifacts_root_path: путь, относительно которого заданы пути артифактов в config. 
    """
    def __init__(
        self,
        engine: Engine,
        config: dict,
        artifacts_root_path: str,
        filenames: Optional[Dict[str, str]] = None,
        logger: logging.Logger = SILENT_LOGGER,
        **kwargs
    ):
        super(CampaignFlow, self).__init__(**kwargs)
        self.engine = engine
        self.config = config
        self.artifacts_root_path = artifacts_root_path
        self._init_filenames(filenames)
        self.logger = logger

        self.date_to = datetime.datetime.strptime(self.config['date_to'], '%Y-%m-%d').date()
        self.random_state = self.config['random_state']
        self.n_clients = sum([group['size'] for group in self.config['groups']])

        self._init_extract_config(self.config['extract'])
        self.transform_pipeline = create_transform_pipeline(
            config=self.config['transform'],
            artifacts_root_path=self.artifacts_root_path
        )

        self.set_stages([
            'select_suitable_clients',
            'extract_features',
            'transform_features',
            'split_clients',
            'recommend',
        ])

    def _init_filenames(self, filenames: Optional[Dict[str, str]] = None):
        filenames = filenames or {}

        for file_name in [
            'clients',
            'raw_features',
            'features',
            'result',
        ]:
            file_path = os.path.join(self.get_directory(), f'{file_name}.parquet')
            self.__dict__[f'{file_name}_path'] =  filenames.get(f'{file_name}_path', file_path)

        for file_name in [
            'client_groups',
        ]:
            file_path = os.path.join(self.get_directory(), f'{file_name}.pkl')
            self.__dict__[f'{file_name}_path'] =  filenames.get(f'{file_name}_path', file_path)

    def _init_extract_config(self, config_path: list):
        self.extract_config = load_json(os.path.join(self.artifacts_root_path, config_path))
        for calcer_config in self.extract_config:
            if issubclass(fcompute.CALCER_REFERENCE[calcer_config['name']], fbase.DateFeatureCalcer): # TODO: fix reference
                calcer_config['args']['date_to'] = self.date_to

    def _run_stage_select_suitable_clients(self):
        """Начальный отбор клиентов в кампанию."""
        self.clients = select_suitable_clients(self.engine, n_clients=self.n_clients, date_to=self.date_to)
        self.clients.to_parquet(self.clients_path)

    def _restore_stage_select_suitable_clients(self):
        self.clients = pd.read_parquet(self.clients_path)

    def _run_stage_extract_features(self):
        """Сбор признаков из источников."""
        self.raw_features = fcompute.extract_features(
            engine=self.engine,
            config=self.extract_config
        )
        self.raw_features.to_parquet(self.raw_features_path)
        self.raw_features = dd.read_parquet(self.raw_features_path)

    def _restore_stage_extract_features(self):
        self.restore_stage(self.get_previous_stage('extract_features'))
        self.raw_features = dd.read_parquet(self.raw_features_path)

    def _run_stage_transform_features(self):
        '''Второй шаг расчета факторов. Происходит преобразование факторов.'''
        _raw_features = dd.from_pandas(self.clients, npartitions=1).merge(
            self.raw_features, on=['client_id'], how='left'
        ).compute()
        self.features = self.transform_pipeline.transform(_raw_features)
        self.features.to_parquet(self.features_path)

    def _restore_stage_transform_features(self):
        self.restore_stage('select_suitable_clients')
        self.features = pd.read_parquet(self.features_path)

    def _run_stage_split_clients(self):
        """Разбиение клиентов на группы."""
        self.client_groups = split_into_groups(
            self.clients,
            group_sizes=[group['size'] for group in self.config['groups']],
            random_state=self.random_state
        )
        dump_pickle(self.client_groups, self.client_groups_path)

    def _restore_stage_split_clients(self):
        self.restore_stage(self.get_previous_stage('split_clients'))
        self.client_groups = load_pickle(self.client_groups_path)

    def _run_stage_recommend(self):
        """Выбор предложения каждому клиенту."""
        result = list()

        for group_idx, group in enumerate(self.client_groups):
            group_params = self.config['groups'][group_idx]

            part = group.copy()
            part['group'] = group_params['name']
            if group_params['treatment_params']['type'] == 'constant':
                part['treatment'] = group_params['treatment_params']['treatment']
            elif group_params['treatment_params']['type'] == 'from_column':
                col_treatment = group_params['treatment_params']['col_treatment']
                part = (
                    part
                    .merge(self.features[['client_id', col_treatment]], on=['client_id'], how='left')
                    .rename(columns={col_treatment: "treatment"})
                )
            else:
                raise NotImplementedError(group_params['treatment_params']['type'])

            result.append(part)
        
        self.result = pd.concat(result).reset_index(drop=True)
        self.result.to_parquet(self.result_path)

    def _restore_stage_recommend(self):
        self.restore_stage(self.get_previous_stage('recommend'))
        self.result = pd.read_parquet(self.result_path)

    def run_stage(self, stage_id: str):
        self.logger.info(f'Run stage {stage_id} {self.stages.index(stage_id) + 1} out of {len(self.stages)}')
        super(CampaignFlow, self).run_stage(stage_id)
        self.logger.info(f'Finished stage {stage_id}')
