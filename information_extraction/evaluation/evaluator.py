from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from information_extraction.data.ground_truths import GROUND_TRUTHS
from information_extraction.data.metrics import normalize_answer
from information_extraction.dtypes import EvaluationResult, SegmentPrediction, DocumentPrediction


class Evaluator:
    def __init__(self, metrics: Dict[str, Callable]):
        self.metrics = metrics

    def evaluate_segments(self, segment_predictions: List[SegmentPrediction]) -> EvaluationResult:
        df_segments = self.segments_to_table(segment_predictions)

        segment_metrics = self.get_segment_metrics(df_segments)

        return EvaluationResult(df_segments, segment_metrics, None)

    def evaluate_documents(self, document_predictions: DocumentPrediction) -> EvaluationResult:
        df_segments = self.segments_to_table(document_predictions.segments)
        df_documents = self.documents_to_table(document_predictions.predictions)

        segment_metrics = self.get_segment_metrics(df_segments)
        document_metrics = self.get_document_metrics(df_documents)

        return EvaluationResult(df_segments, segment_metrics, df_documents, document_metrics)

    def get_segment_metrics(self, df_segments: pd.DataFrame) -> dict:
        return {
            metric_name: df_segments[metric_name].mean()
            for metric_name in self.metrics
        }

    def get_document_metrics(self, df_documents: pd.DataFrame) -> dict:
        return {
            'instance': self.get_instance_metrics(df_documents),
            'global': self.get_global_metrics(df_documents),
        }

    def get_document_summary(self, df_documents: pd.DataFrame) -> pd.Series:
        global_metrics = self.get_global_metrics(df_documents)
        instance_metrics = self.get_instance_metrics(df_documents)
        return pd.Series({
            ('Global', 'P'): global_metrics['precision'],
            ('Global', 'R'): global_metrics['recall'],
            ('Global', 'F1'): global_metrics['f1'],
            ('Instance', 'EM'): instance_metrics['em'],
            ('Instance', 'F1'): instance_metrics['f1'],
        })

    def get_instance_metrics(self, df_documents: pd.DataFrame) -> dict:
        instance_metrics = {}

        for metric_name in self.metrics:
            columns = [column for column in df_documents.columns if column.endswith(f'/{metric_name}')]
            instance_metrics[metric_name] = df_documents[columns].values.mean()

        attributes = [column[:-5] for column in df_documents.columns if column.endswith('/true')]

        for attribute in attributes:
            instance_metrics[attribute] = {
                metric_name: df_documents[f'{attribute}/{metric_name}'].mean()
                for metric_name in self.metrics
            }

        return instance_metrics

    def get_global_metrics(self, df_documents: pd.DataFrame) -> dict:
        global_metrics = {}

        if 'attribute' in df_documents.columns and 'result/true' in df_documents.columns:
            # We compute the metrics per document and macro-average over the documents
            p, r, f1 = map(np.mean, zip(*[
                self.get_precision_recall_f1(df_doc['result/true'], df_doc['result/pred'])
                for _, df_doc in df_documents.groupby('doc_id')
            ]))

            return {
                'precision': p,
                'recall': r,
                'f1': f1
            }

        attributes = [column[:-5] for column in df_documents.columns if column.endswith('/true')]

        for attribute in attributes:
            p, r, f1 = self.get_precision_recall_f1(df_documents[f'{attribute}/true'],
                                                    df_documents[f'{attribute}/pred'])
            global_metrics[attribute] = {
                'precision': p,
                'recall': r,
                'f1': f1
            }

        for metric_name in ['precision', 'recall', 'f1']:
            global_metrics[metric_name] = np.mean([
                global_metrics[attribute][metric_name]
                for attribute in attributes
            ])

        return global_metrics

    @staticmethod
    def get_precision_recall_f1(y_true, y_pred) -> Tuple[float, float, float]:
        tp = fp = tn = fn = 0

        for a_true, a_pred in zip(y_true, y_pred):
            a_true = normalize_answer(a_true)
            a_pred = normalize_answer(a_pred)

            if a_true == '' and a_pred == '':
                tn += 1
            elif a_true == '':
                fp += 1
            elif a_pred == '':
                fn += 1
            elif a_true != a_pred:
                fn += 1
                fp += 1
            else:
                tp += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1

    def segments_to_table(self, segments: List[SegmentPrediction]) -> pd.DataFrame:
        tables = []

        for segment in segments:
            batch = segment.batch
            segment_data = {
                'doc_id': batch.docs,
                'feature': batch.features,
                'context': batch.inputs,
                'expected': batch.targets,
                'predicted': segment.predictions,
                'score': segment.scores,
            }

            for metric_name, metric_fct in self.metrics.items():
                segment_data[metric_name] = [metric_fct(a_true, a_pred)
                                             for a_true, a_pred
                                             in zip(batch.targets, segment.predictions)]

            tables.append(pd.DataFrame(segment_data))

        return pd.concat(tables, ignore_index=True)

    def documents_to_table(self, document_predictions: dict) -> pd.DataFrame:
        documents = []
        for doc_id, predictions in document_predictions.items():
            document = {'doc_id': doc_id}

            for attribute, prediction in predictions.items():
                # Find result that best matches the prediction
                expected = max(GROUND_TRUTHS[doc_id][attribute],
                               key=lambda a_true: tuple(metric_fct(a_true, prediction['prediction'])
                                                        for metric_fct in self.metrics.values()))

                document[f'{attribute}/true'] = expected
                document[f'{attribute}/pred'] = prediction['prediction']
                document[f'{attribute}/conf'] = prediction['confidence']

                for metric_name, metric_fct in self.metrics.items():
                    document[f'{attribute}/{metric_name}'] = metric_fct(expected, prediction['prediction'])

            documents.append(document)

        return pd.DataFrame(documents)
