from tqdm import tqdm

from callbacks.base import BaseCallback
from outputs import EvaluationResult


class EarlyStopping(BaseCallback):
    def __init__(self, patience: int, metric: str = 'loss', mode: str = 'min'):
        self.patience = patience
        self.metric = metric
        self.mode = mode

        if mode not in ('min', 'max'):
            raise ValueError('`mode` must be one of ("min", "max")')

        self.best_score = None
        self.steps_since_improve = 0

    def on_train_start(self, run_params: dict):
        self.steps_since_improve = 0

    def on_validation_end(self, step_num: int, results: EvaluationResult):
        new_score = results.metrics[self.metric]

        if (self.best_score is None or
                (new_score < self.best_score and self.mode == 'min') or
                (new_score > self.best_score and self.mode == 'max')):
            self.best_score = new_score
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1

            if self.steps_since_improve >= self.patience:
                tqdm.write(f'Early stopping: {self.metric} did not improve for {self.steps_since_improve} steps...')
                self.trainer.stop()
