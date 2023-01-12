from collections import Counter
import sys
from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler

from information_extraction.training.data.base import BaseDataset


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
        return sys.maxsize if self.num_samples is None else self.num_samples
