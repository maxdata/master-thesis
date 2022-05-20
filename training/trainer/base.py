from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import PreTrainedTokenizer
from transformers import logging

import torch
from torch.nn import Module
from torch.utils.data import DataLoader


from callbacks import BaseCallback

logging.set_verbosity_error()


class BaseTrainer(ABC):
    architecture = ''

    def __init__(
        self,
        model: Module,
        tokenizer: PreTrainedTokenizer,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        validate_per_steps: int = -1,
        metrics: Optional[Dict[str, Callable]] = None,
        device: Optional[str] = None,
        learning_rate: Optional[float] = 5e-5,
        optimizer: str = 'adamw',
        callbacks: Optional[List[BaseCallback]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.validate_per_steps = validate_per_steps
        self.metrics = metrics or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.callbacks = callbacks or []
        self.is_training = False

        self.do_validate = self.validate_per_steps > 0
        self.model.to(self.device)

        optimizers = {
            'adamw': torch.optim.AdamW
        }

        self.optimizer = optimizers[self.optimizer_name](self.model.parameters(), lr=self.learning_rate)

        if self.do_validate and val_loader is None:
            raise ValueError('`val_loader` must be provided if `validate_per_steps` > 0')

    def train(self, num_steps: int, batch_size: int):
        if self.train_loader is None:
            raise ValueError('`train_loader` must be provided if model needs training')

        assert batch_size % self.train_loader.batch_size == 0, f'Batch size must be a multiple of {self.train_loader.batch_size}'

        grad_accumulation_steps = batch_size // self.train_loader.batch_size

        self.on_train_start({
            'num_steps': num_steps,
            'batch_size': batch_size,
            'architecture': self.architecture,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer_name,
        })

        data_iterator = iter(self.train_loader)

        pbar = tqdm(total=num_steps, desc='Training')

        # Perform a validation step to have proper nice graphs
        if self.do_validate:
            metrics = self.evaluate(self.val_loader, label='Validating')
            self.on_validation_end(0, metrics, num_steps)

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

            if self.do_validate and step_num % self.validate_per_steps == 0:
                metrics = self.evaluate(self.val_loader, label='Validating')

                self.on_validation_end(step_num, metrics, num_steps)

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

    def on_validation_end(self, step_num, metrics, total_steps):
        formatted = ', '.join(f'val/{metric}: {value:.2f}' for metric, value in metrics.items())
        num = str(step_num).rjust(len(str(total_steps)))

        tqdm.write(f'[Step {num}] {formatted}')

        for callback in self.callbacks:
            callback.on_validation_end(step_num, metrics)

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    def evaluate(self, dataloader: DataLoader, report_instances: bool = False,
                 report_documents: bool = False, label: Optional[str] = None):
        self.model.eval()

        losses = []
        full_results = []

        if label is None:
            iterator = dataloader
        else:
            iterator = tqdm(dataloader, desc=label, leave=False)

        for batch in iterator:
            loss, predictions, scores = self.evaluate_step(batch)

            losses.append(loss)

            instance_data = {
                'doc_id': batch.docs,
                'feature': batch.features,
                'context': batch.inputs,
                'expected': batch.targets,
                'predicted': predictions,
                'score': scores,
            }

            for metric_name, metric_fct in self.metrics.items():
                instance_data[metric_name] = [metric_fct(a_true, a_pred)
                                              for a_true, a_pred
                                              in zip(batch.targets, predictions)]

            full_results.append(pd.DataFrame(instance_data))

        instances = pd.concat(full_results, ignore_index=True)
        documents = self.segments_to_documents(instances)

        self.model.train()

        agg_metrics = {
            'loss': np.mean(losses),
        }
        for metric_name in self.metrics:
            agg_metrics[f'instance/{metric_name}'] = instances[metric_name].mean()
            agg_metrics[f'document/{metric_name}'] = documents[metric_name].mean()

            for feature in instances['feature'].unique():
                agg_metrics[f'document/{feature}/{metric_name}'] = documents[f'{feature}/{metric_name}'].mean()

        if not report_instances and not report_documents:
            return agg_metrics

        results = {
            'agg': agg_metrics
        }

        if report_instances:
            results['instances'] = instances

        if report_documents:
            results['documents'] = documents

        return results

    def perform_evaluation(self, **loaders: DataLoader):
        results = {}

        for key, loader in loaders.items():
            results[key] = self.evaluate(loader, report_instances=True, report_documents=True,
                                         label=f'Evaluating "{key}"')

        for callback in self.callbacks:
            callback.on_evaluation_end(results)

        return results

    @abstractmethod
    def evaluate_step(self, batch) -> Tuple[float, Iterable[str], Iterable[float]]:
        """Performs evaluation for a single batch. Returns the loss, a list of predictions, and a list of scores"""
        raise NotImplementedError

    def segments_to_documents(self, instances: pd.DataFrame) -> pd.DataFrame:
        instances = instances[~((instances['predicted'] == '') | pd.isnull(instances['predicted']))]

        results = []

        for doc_id, doc_instances in instances.groupby('doc_id'):
            doc_predictions = {'doc_id': doc_id}

            scores = defaultdict(list)

            for feature, predictions in doc_instances.groupby('feature'):
                best_prediction = predictions.loc[predictions['score'].idxmax()]

                doc_predictions[feature] = best_prediction['predicted']
                doc_predictions[f'{feature}/score'] = best_prediction['score']

                for metric_name in self.metrics:
                    doc_predictions[f'{feature}/{metric_name}'] = best_prediction[metric_name]
                    scores[metric_name].append(best_prediction[metric_name])

            for metric_name, metric_scores in scores.items():
                doc_predictions[metric_name] = np.mean(metric_scores)

            results.append(doc_predictions)

        return pd.DataFrame(results)

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
