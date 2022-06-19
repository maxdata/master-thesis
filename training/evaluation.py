from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader

from data.ground_truths import GroundTruths
from outputs import EvaluationResult, SegmentPrediction
from trainer import BaseTrainer


class Evaluator:
    def __init__(
        self,
        trainer: BaseTrainer,
        metrics: Dict[str, Callable],
        ground_truths: Optional[GroundTruths] = None,
    ):
        self.trainer = trainer
        self.metrics = metrics
        self.ground_truths = ground_truths

    def evaluate_segments(self, dataloader: DataLoader, label: Optional[str] = None):
        self.trainer.model.eval()

        losses = []
        segments = []

        if label is None:
            iterator = dataloader
        else:
            iterator = tqdm(dataloader, desc=label, leave=False)

        for batch in iterator:
            loss, segment_outputs = self.trainer.predict_segment_batch(batch)

            losses.append(loss)
            segments.append(segment_outputs)

        self.trainer.model.train()

        df_segments = self.segments_to_table(segments)

        agg_metrics = {
            'segment/loss': np.mean(losses),
        }
        for metric_name in self.metrics:
            agg_metrics[f'segment/{metric_name}'] = df_segments[metric_name].mean()

        return EvaluationResult(agg_metrics, df_segments, None)

    def evaluate_documents(self, loader: DataLoader, label: Optional[str] = None):
        if self.ground_truths is None:
            raise ValueError('Cannot perform document evaluation if ground truths are not provided!')

        if label is None:
            iterator = loader
        else:
            iterator = tqdm(loader, desc=label, leave=False)

        # TODO: change method to greedy
        document_predictions, segments = self.trainer.predict_documents(iterator)

        documents = []
        for doc_id, predictions in document_predictions.items():
            document = {'doc_id': doc_id}

            for attribute, prediction in predictions.items():
                # Find result that best matches the prediction
                expected = max(self.ground_truths[doc_id][attribute],
                               key=lambda a_true: tuple(metric_fct(a_true, prediction['prediction'])
                                                        for metric_fct in self.metrics.values()))

                document[f'{attribute}/true'] = expected
                document[f'{attribute}/pred'] = prediction['prediction']
                document[f'{attribute}/conf'] = prediction['confidence']

                for metric_name, metric_fct in self.metrics.items():
                    document[f'{attribute}/{metric_name}'] = metric_fct(expected, prediction['prediction'])

            documents.append(document)

        df_segments = self.segments_to_table(segments)
        df_documents = pd.DataFrame(documents)

        agg_metrics = {}

        for metric_name in self.metrics:
            agg_metrics[f'segment/{metric_name}'] = df_segments[metric_name].mean()

        for metric_name in self.metrics:
            columns = [column for column in df_documents.columns if column.endswith(f'/{metric_name}')]
            agg_metrics[f'document/{metric_name}'] = df_documents[columns].values.mean()

        for column in df_documents.columns:
            for metric_name in self.metrics:
                if column.endswith(f'/{metric_name}'):
                    agg_metrics[f'document/{column}'] = df_documents[column].mean()

        return EvaluationResult(agg_metrics, df_segments, df_documents)

    def segments_to_table(self, segments: List[SegmentPrediction]) -> pd.DataFrame:
        tables = []

        for batch, predictions, scores in segments:
            segment_data = {
                'doc_id': batch.docs,
                'feature': batch.features,
                'context': batch.inputs,
                'expected': batch.targets,
                'predicted': predictions,
                'score': scores,
            }

            for metric_name, metric_fct in self.metrics.items():
                segment_data[metric_name] = [metric_fct(a_true, a_pred)
                                             for a_true, a_pred
                                             in zip(batch.targets, predictions)]

            tables.append(pd.DataFrame(segment_data))

        return pd.concat(tables, ignore_index=True)
