import pandas as pd
import pickle
import os
import subprocess

from typing import List


class Flow:

    def __init__(
        self,
        run_id: str,
        runs_root_path: str,
        stages: List[str] = None,
    ):
        self.run_id = run_id
        self.directory = os.path.join(runs_root_path, run_id)
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        if stages is not None:
            self.set_stages(stages)

    def set_stages(self, stages: List[str]):
        self.stages = stages

    def get_directory(self):
        return self.directory

    def commit_stage(self, stage_id: str):
        subprocess.run(f'touch "{self.get_directory()}/_stage_{stage_id}_completed.flg"', shell=True)

    def detect_last_successfull_stage(self) -> str:
        result = None
        for stage_id in self.stages:
            result = stage_id if os.path.exists(f'{self.get_directory()}/_stage_{stage_id}_completed.flg') else result
        return result

    def restore_stage(self, stage_id: str):
        self.__getattribute__(f'_restore_stage_{stage_id}')()

    def get_stage_num(self, stage_id: str) -> int:
        return -1 if stage_id is None else self.stages.index(stage_id)

    def get_previous_stage(self, stage_id: str) -> int:
        stage_num = self.get_stage_num(stage_id)
        return None if stage_num <= 0 else self.stages[stage_num - 1]

    def run_stage(self, stage_id: str):
        self.__getattribute__(f'_run_stage_{stage_id}')()
        self.commit_stage(stage_id)

    def run(self):
        last_successfull_stage = self.detect_last_successfull_stage()
        if last_successfull_stage is not None:
            self.restore_stage(last_successfull_stage)

        last_successfull_stage_num = self.get_stage_num(last_successfull_stage)
        for stage_id in self.stages[last_successfull_stage_num + 1: ]:
            self.run_stage(stage_id)
