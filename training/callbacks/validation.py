from typing import Optional, Union

from torch.utils.data import DataLoader
from tqdm import tqdm

from callbacks.base import BaseCallback
from evaluation import Evaluator


class ValidationCallback(BaseCallback):
    def __init__(
        self,
        evaluator: Evaluator,
        validation_interval: Union[int, float],
        segment_loader: Optional[DataLoader] = None,
        document_loader: Optional[DataLoader] = None,
        label: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.validation_interval = validation_interval
        self.segment_loader = segment_loader
        self.document_loader = document_loader
        self.label = label
        self.total_steps = 0
        self.document_method = 'greedy'

        if segment_loader and document_loader:
            raise ValueError('Only one of `segment_loader` and `document_loader` can be provided...')

        if segment_loader is not None:
            self.method = 'segments'
        elif document_loader is not None:
            self.method = 'documents'
        else:
            raise ValueError('At least one of `segment_loader` and `document_loader` must be provided...')

    def on_train_start(self, run_params: dict):
        self.total_steps = run_params['num_steps']

        if run_params['model_name'] == 'Linear':
            self.document_method = 'rerank'
        else:
            self.document_method = 'greedy'

        self.perform_validation(0)

    def on_step_end(self, step_num: int, loss: float):
        if 0 < self.validation_interval < 1:
            interval = round(self.validation_interval * self.total_steps)
        else:
            interval = round(self.validation_interval)

        if step_num % interval == 0:
            self.perform_validation(step_num)

    def perform_validation(self, step_num: int):
        if self.method == 'segments':
            results = self.evaluator.evaluate_segments(self.segment_loader, label=self.label)
        else:
            results = self.evaluator.evaluate_documents(self.document_loader, method=self.document_method,
                                                        label=self.label)

        formatted = ', '.join(f'val/{metric}: {value:.2f}' for metric, value in results.metrics.items())
        num = str(step_num).rjust(len(str(self.total_steps)))

        tqdm.write(f'[Step {num}] {formatted}')

        for callback in self.trainer.callbacks:
            callback.on_validation_end(step_num, results)
