from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from transformers import PreTrainedTokenizer
from transformers import logging

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


from callbacks import BaseCallback
from outputs import DocumentPrediction, SegmentPrediction

logging.set_verbosity_error()


class BaseTrainer(ABC):
    architecture = ''

    def __init__(
        self,
        model: Module,
        tokenizer: PreTrainedTokenizer,
        train_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
        learning_rate: Optional[float] = 5e-5,
        optimizer: str = 'adamw',
        callbacks: Optional[List[BaseCallback]] = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.callbacks = callbacks or []
        self.model_kwargs = kwargs

        self.is_training = False
        self.model.to(self.device)

        optimizers = {
            'adamw': torch.optim.AdamW
        }

        self.optimizer = optimizers[self.optimizer_name](self.model.parameters(), lr=self.learning_rate)

    def train(self, num_steps: int, batch_size: int):
        if self.train_loader is None:
            raise ValueError('`train_loader` must be provided if model needs training')

        mini_batch_size = self.train_loader.batch_size or getattr(self.train_loader.sampler, 'batch_size')

        if mini_batch_size is None:
            print('Warning: could not recognize mini-batch size. Disabling gradient accumulation...')
            mini_batch_size = batch_size
        else:
            assert batch_size % mini_batch_size == 0, f'Batch size must be a multiple of {mini_batch_size}'

        grad_accumulation_steps = batch_size // mini_batch_size

        self.on_train_start({
            'num_steps': num_steps,
            'batch_size': batch_size,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer_name,
        })

        data_iterator = iter(self.train_loader)

        pbar = tqdm(total=num_steps, desc='Training')

        for step_num in range(1, num_steps + 1):
            losses = []

            for _ in range(grad_accumulation_steps):
                batch = next(data_iterator)

                loss = self.train_step(batch)
                loss.backward()

                losses.append(float(loss))

            self.optimizer.step()
            self.optimizer.zero_grad()

            pbar.update()
            pbar.set_postfix({'loss': np.mean(losses)})

            self.on_step_end(step_num, np.mean(losses))

            if not self.is_training:
                # Training was stopped, most likely by early stopping
                pbar.close()
                break

        self.on_train_end()

    def on_train_start(self, run_params: dict):
        print(f'Training {self.model.__class__.__name__} for {run_params["num_steps"]} steps on device `{self.device}`')
        self.is_training = True
        self.optimizer.zero_grad()

        for callback in self.callbacks:
            callback.on_train_start(run_params)

    def on_train_end(self):
        self.is_training = False
        for callback in self.callbacks:
            callback.on_train_end()

    def on_step_end(self, step_num, loss):
        for callback in self.callbacks:
            callback.on_step_end(step_num, loss)

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def predict_segment_batch(self, batch) -> Tuple[float, SegmentPrediction]:
        """Performs inference for a batch of segments. Returns the loss, a list of predictions, and a list of scores"""
        raise NotImplementedError

    @abstractmethod
    def predict_document_batch(self, dataloaders: Iterable[Tuple[str, DataLoader]],
                               method: str = 'greedy') -> DocumentPrediction:
        """Evaluates a single document. Returns, for each attribute, the prediction and confidence."""
        raise NotImplementedError

    def stop(self):
        self.is_training = False

    def init(self):
        for callback in self.callbacks:
            callback.on_trainer_init(self)

    def finish(self):
        for callback in self.callbacks:
            callback.on_trainer_finish()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
