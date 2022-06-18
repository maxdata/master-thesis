from collections import Counter, defaultdict
import random
import sys
from typing import List, Iterator, Optional, Tuple

import torch
from torch.utils.data import Sampler, BatchSampler

from data.base import BaseDataset


class SegmentSampler(Sampler[int]):
    def __init__(self, data_source: BaseDataset, num_samples: Optional[int] = None,
                 replacement: bool = True, remove_null: bool = False, generator=None) -> None:
        self.data_source = data_source
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.remove_null = remove_null

        self.batch_size = self.num_samples or 32

        feature_counts = Counter(self.data_source.features)
        self.weights = torch.as_tensor([1 / feature_counts[feature] for feature in self.data_source.features], dtype=torch.float)
        self.value_weights = self.weights[self.data_source.indices_with_data]
        self.null_weights = self.weights[self.data_source.indices_without_data]

    def single_iteration(self):
        if self.remove_null:
            rand_tensor = torch.multinomial(self.weights, self.batch_size, self.replacement, generator=self.generator)
            yield from iter(rand_tensor.tolist())
        else:
            value_rand_tensor = torch.multinomial(self.value_weights, self.batch_size // 2, self.replacement,
                                                  generator=self.generator)
            null_rand_tensor = torch.multinomial(self.null_weights, self.batch_size // 2, self.replacement,
                                                 generator=self.generator)

            for value_index, null_index in zip(value_rand_tensor, null_rand_tensor):
                yield self.data_source.indices_with_data[value_index]
                yield self.data_source.indices_without_data[null_index]

    def __iter__(self) -> Iterator[int]:
        if self.num_samples is None:
            while True:
                yield from self.single_iteration()
        else:
            yield from self.single_iteration()

    def __len__(self) -> int:
        if self.num_samples is None:
            return sys.maxsize
        else:
            return self.num_samples


class DocumentSampler(Sampler[Tuple[str, Iterator[Tuple[str, List[int]]]]]):
    """
    Samples from the dataset on a per-document basis. Ensures the segments for each document are split into
    batches of a specified batch size.
    """
    def __init__(self, data_source: BaseDataset, num_documents: Optional[int] = None, batch_size: int = 1,
                 shuffle: bool = False, replacement: bool = False, remove_null: bool = False, generator=None) -> None:
        self.data_source = data_source
        self.num_documents = num_documents
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.replacement = replacement
        self.remove_null = remove_null
        self.generator = generator

        self.document_indices = defaultdict(lambda: defaultdict(list))
        indices = self.data_source.indices_with_data if remove_null else range(len(self.data_source))

        for i in indices:
            self.document_indices[self.data_source.docs[i]][self.data_source.features[i]].append(i)

    def document_iterator(self, doc_id: str) -> Iterator[Tuple[str, List[int]]]:
        for attribute, indices in self.document_indices[doc_id].items():
            batch_sampler = BatchSampler(indices, self.batch_size, drop_last=False)
            yield attribute, list(batch_sampler)

    def __iter__(self) -> Iterator[Tuple[str, Iterator[Tuple[str, List[int]]]]]:
        documents = list(self.document_indices)
        num_returned = 0

        while True:
            if self.shuffle:
                random.shuffle(documents)

            for doc_id in documents:
                if self.num_documents is not None and num_returned >= self.num_documents:
                    return

                yield doc_id, self.document_iterator(doc_id)

                num_returned += 1

            if not self.replacement:
                return

    def __len__(self) -> int:
        if self.num_documents is None:
            if self.replacement:
                return sys.maxsize
            else:
                return len(self.data_source)
        else:
            if self.replacement:
                return self.num_documents
            else:
                return min(self.num_documents, len(self.data_source))
