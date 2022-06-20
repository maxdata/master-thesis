from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from transformers import PreTrainedTokenizer, get_constant_schedule_with_warmup
from transformers import logging

import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate
import torch.nn.functional as F


from callbacks import BaseCallback
from metrics import compute_f1
from outputs import DocumentPrediction, SegmentPrediction

logging.set_verbosity_error()

RerankBatch = namedtuple('RerankBatch', ['docs', 'attributes', 'X', 'y', 'predictions'])
RerankResult = namedtuple('RerankResult', ['doc_id', 'attribute', 'prediction', 'confidence'])


class BaseTrainer(ABC):
    architecture = ''
    optimizers = {
        'adamw': torch.optim.AdamW
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        reranker: Optional[nn.Module] = None,
        segment_loader: Optional[DataLoader] = None,
        document_loader: Optional[DataLoader] = None,
        device: Optional[str] = None,
        learning_rate: Optional[float] = 5e-5,
        optimizer: str = 'adamw',
        callbacks: Optional[List[BaseCallback]] = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reranker = reranker
        self.segment_loader = segment_loader
        self.document_loader = document_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.model_kwargs = kwargs

        self.is_training = False
        self.model.to(self.device)

    def train(
        self,
        num_segment_steps: Optional[int] = None,
        num_document_steps: Optional[int] = None,
        batch_size: int = 32,
        mini_batch_size: Optional[int] = None,
    ):
        if num_segment_steps is not None:
            if self.segment_loader is None:
                raise ValueError('`segment_loader` must be provided if model needs segment training')
            if self.model is None:
                raise ValueError('`model` must be provided if model needs segment training')
        if num_document_steps is not None:
            if self.document_loader is None:
                raise ValueError('`document_loader` must be provided if model needs document training')
            if self.reranker is None:
                raise ValueError('`reranker` must be provided if model needs document training')

        if mini_batch_size is None:
            mini_batch_size = batch_size
        else:
            assert batch_size % mini_batch_size == 0, f'Batch size must be a multiple of {mini_batch_size}'

        self.on_train_start({
            'num_segment_steps': num_segment_steps,
            'num_document_steps': num_document_steps,
            'batch_size': batch_size,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
        })

        if num_segment_steps is not None:
            self.train_module(self.model, self.segment_loader, self.train_step,
                              num_segment_steps, batch_size, mini_batch_size, label='Training base model')
        if num_document_steps is not None:
            self.train_module(self.reranker,
                              self.get_rerank_batches(
                                  self.collect_document_segments(self.document_loader),
                                  mini_batch_size,
                                  topk=self.model_kwargs.get('topk', 1),
                                  length=self.model_kwargs.get('sequence_length', 1),
                              ),
                              self.train_rerank_step, num_document_steps, batch_size, mini_batch_size,
                              label='Training reranker')

        self.on_train_end()

    def train_module(self, model: nn.Module, dataloader: Iterable, step_fct: Callable,
                     num_steps: int, batch_size: int, mini_batch_size: int,
                     label: Optional[str] = None):
        self.is_training = True

        grad_accumulation_steps = batch_size // mini_batch_size

        optimizer = self.optimizers[self.optimizer](model.parameters(), lr=self.learning_rate)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_steps // 50, num_steps)

        data_iterator = iter(dataloader)

        pbar = tqdm(total=num_steps, desc=label) if label is not None else None

        for step_num in range(1, num_steps + 1):
            losses = []

            for _ in range(grad_accumulation_steps):
                batch = next(data_iterator)
                loss = step_fct(batch)
                loss.backward()

                losses.append(float(loss))

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if pbar is not None:
                pbar.update()
                pbar.set_postfix({'loss': np.mean(losses)})

            self.on_step_end(step_num, np.mean(losses))

            if not self.is_training:
                # Training was stopped, most likely by early stopping
                if pbar is not None:
                    pbar.close()
                break

    @abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def train_rerank_step(self, batch: RerankBatch) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss()

        outputs = self.reranker(batch.X)
        return loss_fct(outputs, batch.y)

    def get_rerank_batches(self, segments_per_document: Iterable[List[SegmentPrediction]], mini_batch_size: int,
                           topk: int, length: int) -> Iterator[RerankBatch]:
        batch = []

        for document_segments in segments_per_document:
            batch.append(self.get_rerank_features(document_segments, topk, length))

            if len(batch) == mini_batch_size:
                yield default_collate(batch)
                batch = []

        if batch:
            yield default_collate(batch)

    @staticmethod
    def get_rerank_features(segments: List[SegmentPrediction], topk: int, length: int) -> RerankBatch:
        entries = sorted((
            (score, segment_index, batch_index)
            for segment_index, segment in enumerate(segments)
            for batch_index, score in enumerate(segment.scores)
            if segment.predictions[batch_index]
        ), reverse=True)[:topk]

        if len(entries) < topk:
            entries += sorted((
                (score, segment_index, batch_index)
                for segment_index, segment in enumerate(segments)
                for batch_index, score in enumerate(segment.scores)
                if not segment.predictions[batch_index]
            ))[:topk - len(entries)]

        embeddings = []
        scores = []
        predictions = []

        for _, segment_index, batch_index in entries:
            embeddings.append(segments[segment_index].embeddings[batch_index])
            scores.append(compute_f1(
                segments[segment_index].batch.targets[batch_index],
                segments[segment_index].predictions[batch_index],
            ))
            predictions.append(segments[segment_index].predictions[batch_index])

        doc_id = segments[0].batch.docs[0]
        attribute = segments[0].batch.features[0]

        features = torch.concat([
            torch.flatten(F.pad(embedding.T, (0, length - embedding.shape[0])).T)
            for embedding in embeddings
        ], dim=0)

        best_prediction = max(range(topk), key=lambda i: scores[i])

        return RerankBatch(doc_id, attribute, features, best_prediction, predictions)

    @abstractmethod
    def predict_segment_batch(self, batch) -> Tuple[float, SegmentPrediction]:
        """Performs inference for a batch of segments. Returns the loss, a list of predictions, and a list of scores"""
        raise NotImplementedError

    def collect_document_segments(self, dataloader: DataLoader) -> Iterator[List[SegmentPrediction]]:
        document_segments = []
        current_document = None

        for batch in dataloader:
            new_document = batch.docs[0], batch.features[0]

            if current_document != new_document:
                if current_document is not None and document_segments:
                    yield document_segments

                document_segments = []
                current_document = new_document

            _, segment_prediction = self.predict_segment_batch(batch)
            document_segments.append(segment_prediction)

        if current_document is not None and document_segments:
            yield document_segments

    def predict_documents(self, dataloader: DataLoader, method: str = 'greedy') -> DocumentPrediction:
        documents = defaultdict(dict)
        segments = []

        if method == 'greedy':
            for document_segments in self.collect_document_segments(dataloader):
                segments.extend(document_segments)

                doc_id = document_segments[0].batch.docs[0]
                attribute = document_segments[0].batch.features[0]

                documents[doc_id][attribute] = self.greedy_prediction(document_segments)
        elif method == 'rerank':
            for rerank_batch in self.get_rerank_batches(
                                  self.collect_document_segments(self.document_loader),
                                  32,
                                  topk=self.model_kwargs.get('topk', 1),
                                  length=self.model_kwargs.get('sequence_length', 1),
                              ):
                results = self.rerank_prediction(rerank_batch)

                for result in results:
                    documents[result.doc_id][result.attribute] = {
                        'prediction': result.prediction,
                        'confidence': result.confidence,
                    }

        return DocumentPrediction(documents, segments)

    @staticmethod
    def greedy_prediction(segments: List[SegmentPrediction]) -> dict:
        predictions = []
        scores = []

        for segment in segments:
            predictions.extend(segment.predictions)
            scores.extend(segment.scores)

        if any(predictions):
            score, prediction = max((score, pred) for score, pred in zip(scores, predictions) if pred)
        else:
            score, prediction = min(zip(scores, predictions))
            score = 1 - score

        return {
            'prediction': prediction,
            'confidence': score,
        }

    def rerank_prediction(self, batch: RerankBatch) -> List[RerankResult]:
        with torch.no_grad():
            outputs = F.softmax(self.reranker(batch.X), dim=-1)
            best_indices, scores = outputs.max(dim=-1)

        results = []
        for i, (best_index, score) in enumerate(zip(best_indices, scores)):
            results.append(RerankResult(batch.docs[i], batch.attributes[i], batch.predictions[i][best_index], score))

        return results

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
