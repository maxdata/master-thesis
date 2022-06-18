import json
import os
import sys
from typing import Optional

from callbacks.base import BaseCallback
from outputs import EvaluationResult


class LoggingCallback(BaseCallback):
    def __init__(self, train_log: Optional[str] = None, val_log: Optional[str] = None):
        self.train_log = train_log
        self.val_log = val_log

        if self.train_log and os.path.exists(self.train_log):
            print(f'Warning: file {self.train_log} already exists!', file=sys.stderr)

        if self.val_log and os.path.exists(self.val_log):
            print(f'Warning: file {self.val_log} already exists!', file=sys.stderr)

    def on_step_end(self, step_num: int, loss: float):
        if self.train_log is None:
            return

        with open(self.train_log, 'a') as _file:
            json.dump({'step_num': step_num, 'loss': loss}, _file)
            _file.write('\n')

    def on_validation_end(self, step_num: int, results: EvaluationResult):
        if self.val_log is None:
            return

        to_log = {'step_num': step_num, **results.metrics}
        with open(self.val_log, 'a') as _file:
            json.dump(to_log, _file)
            _file.write('\n')
