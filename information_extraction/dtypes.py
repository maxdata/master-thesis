from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch

PathLike = Union[str, Path]

T5Batch = namedtuple('T5Batch', ['docs', 'inputs', 'targets', 'features',
                                 'input_ids', 'attention_mask', 'decoder_input_ids',
                                 'decoder_attention_mask', 'target_labels'])

BertBatch = namedtuple('BertBatch', ['docs', 'inputs', 'html_embeddings', 'targets', 'features',
                                     'input_ids', 'attention_mask', 'token_type_ids',
                                     'start_positions', 'end_positions'])

Batch = Union[BertBatch, T5Batch]


@dataclass
class SegmentPrediction:
    batch: Batch
    predictions: List[str]
    scores: List[float]
    embeddings: Optional[torch.Tensor] = None


@dataclass
class DocumentPrediction:
    predictions: dict
    segments: List[SegmentPrediction]


@dataclass
class EvaluationResult:
    segments: pd.DataFrame
    segment_metrics: dict

    documents: Optional[pd.DataFrame] = None
    document_metrics: Optional[dict] = None

    @property
    def flattened_metrics(self) -> Dict[str, Any]:
        def flatten(metrics, prefix=''):
            if metrics is None:
                return []
            elif isinstance(metrics, dict):
                results = []
                for key, values in metrics.items():
                    new_prefix = f'{prefix}/{key}' if prefix else key
                    results.extend(flatten(values, new_prefix))
                return results
            else:
                return [(prefix, metrics)]

        return dict(flatten({'segment': self.segment_metrics, 'document': self.document_metrics}))
