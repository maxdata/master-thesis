from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from transformers import get_linear_schedule_with_warmup, logging, PreTrainedTokenizer

import torch
from torch import nn
from torch.utils.data import DataLoader

from information_extraction.dtypes import DocumentPrediction, SegmentPrediction
from information_extraction.training.callbacks import BaseCallback

logging.set_verbosity_error()


class BaseTrainer(ABC):
    architecture = ''
    optimizers = {
        'adamw': torch.optim.AdamW
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        learning_rate: Optional[float] = 5e-5,
        optimizer: str = 'adamw',
        callbacks: Optional[List[BaseCallback]] = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.model_kwargs = kwargs

        self.is_training = False
        self.model.to(self.device)

    def train(self, dataloader: DataLoader, num_steps: int, batch_size: int, warmup_steps: int = 0):
        mini_batch_size = dataloader.batch_size or getattr(dataloader.sampler, 'batch_size')

        if mini_batch_size is None:
            print('Warning: could not recognize mini-batch size. Disabling gradient accumulation...')
            mini_batch_size = batch_size
        else:
            assert batch_size % mini_batch_size == 0, f'Batch size must be a multiple of {mini_batch_size}'

        grad_accumulation_steps = batch_size // mini_batch_size

        optimizer = self.optimizers[self.optimizer](self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_steps)

        self.on_train_start({
            'num_steps': num_steps,
            'batch_size': batch_size,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
        })

        data_iterator = iter(dataloader)

        pbar = tqdm(total=num_steps, desc='Training')

        for step_num in range(1, num_steps + 1):
            losses = []

            for _ in range(grad_accumulation_steps):
                batch = next(data_iterator)

                loss = self.train_step(batch)
                loss.backward()

                losses.append(float(loss))

            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            pbar.update()
            pbar.set_postfix({'loss': np.mean(losses)})

            self.on_step_end(step_num, np.mean(losses))

            if not self.is_training:
                # Training was stopped, most likely by early stopping
                pbar.close()
                break

        self.on_train_end()

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def predict_segment_batch(self, batch) -> Tuple[float, SegmentPrediction]:
        """Performs inference for a batch of segments. Returns the loss, a list of predictions, and a list of scores"""
        raise NotImplementedError

    def predict_segments(self, dataloader: Iterable) -> List[SegmentPrediction]:
        segments = []

        for batch in dataloader:
            loss, segment_outputs = self.predict_segment_batch(batch)
            segments.append(segment_outputs)

        return segments

    def predict_documents(self, dataloader: Iterable, method: str = 'greedy') -> DocumentPrediction:
        documents = defaultdict(dict)
        segments = []

        document_segments = []
        current_document = None

        for batch in dataloader:
            new_document = batch.docs[0], batch.features[0]

            if current_document != new_document:
                if current_document is not None and document_segments:
                    doc_id, attribute = current_document
                    documents[doc_id][attribute] = self.segments_to_prediction(document_segments, method=method)
                document_segments = []
                current_document = new_document

            _, segment_prediction = self.predict_segment_batch(batch)
            segments.append(segment_prediction)
            document_segments.append(segment_prediction)

        if current_document is not None and document_segments:
            doc_id, attribute = current_document
            documents[doc_id][attribute] = self.segments_to_prediction(document_segments, method=method)

        return DocumentPrediction(documents, segments)

    @staticmethod
    def segments_to_prediction(segments: List[SegmentPrediction], method: str = 'greedy') -> dict:
        predictions = []
        scores = []

        for segment in segments:
            predictions.extend(segment.predictions)
            scores.extend(segment.scores)

        if method != 'greedy':
            # TODO: implement new method
            raise ValueError(f'Prediction method `{method}` does not exist!')

        if any(predictions):
            score, prediction = max((score, pred) for score, pred in zip(scores, predictions) if pred)
        else:
            score, prediction = min(zip(scores, predictions))
            score = 1 - score
        return {
            'prediction': prediction,
            'confidence': score,
        }

    def on_train_start(self, run_params: dict):
        print(f'Training {self.model.__class__.__name__} for {run_params["num_steps"]} steps on device `{self.device}`')
        self.is_training = True

        for callback in self.callbacks:
            callback.on_train_start(run_params)

    def on_train_end(self):
        self.is_training = False
        for callback in self.callbacks:
            callback.on_train_end()

    def on_step_end(self, step_num, loss):
        for callback in self.callbacks:
            callback.on_step_end(step_num, loss)

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
