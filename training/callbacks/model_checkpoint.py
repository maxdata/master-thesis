import os
from typing import Dict

import torch
from tqdm import tqdm

from callbacks.base import BaseCallback
from outputs import EvaluationResult


class ModelCheckpoint(BaseCallback):
    def __init__(self, filename: str, metric: str = 'loss', mode: str = 'min', restore: bool = False):
        self.filename = filename
        self.metric = metric
        self.mode = mode
        self.restore = restore

        if mode not in ('min', 'max'):
            raise ValueError('`mode` must be one of ("min", "max")')

        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)

        self.best_score = None

    def on_validation_end(self, step_num: int, results: EvaluationResult):
        new_score = results.metrics[self.metric]

        if (self.best_score is None or
                (new_score < self.best_score and self.mode == 'min') or
                (new_score > self.best_score and self.mode == 'max')):
            tqdm.write(f'Saving new best model to {self.filename}...')

            self.best_score = new_score
            torch.save(self.trainer.model.state_dict(), self.filename)

    def on_train_end(self):
        if self.restore:
            tqdm.write(f'Restoring best model checkpoint from {self.filename}...')
            self.trainer.model.load_state_dict(torch.load(self.filename))
