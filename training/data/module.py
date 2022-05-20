from pathlib import Path
from typing import Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler
from transformers import BertTokenizerFast, T5TokenizerFast

from data.base import BaseDataset
from data.bert import BertDataset
from data.sampler import SWDESampler
from data.t5 import T5Dataset

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


class SWDEDataModule:
    def __init__(self,
                 model_version: str,
                 max_length: int,
                 train_files: Optional[List[Union[str, Path]]] = None,
                 val_files: Optional[List[Union[str, Path]]] = None,
                 test_files: Optional[List[Union[str, Path]]] = None,
                 batch_size: int = 32,
                 num_workers: int = 2,
                 remove_null: bool = False):
        super().__init__()
        self.model_version = model_version
        self.max_length = max_length
        self.train_files = train_files or []
        self.val_files = val_files or []
        self.test_files = test_files or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.remove_null = remove_null

        if 'bert' in model_version.lower():
            self.tokenizer = BertTokenizerFast.from_pretrained(model_version)
            self.dataset_class = BertDataset
        elif 't5' in model_version.lower():
            self.tokenizer = T5TokenizerFast.from_pretrained(model_version)
            self.dataset_class = T5Dataset
        else:
            raise ValueError(f'Unsupported model version: {model_version}')

        self.data_train, self.data_val, self.data_test = self.prepare_datasets()

    def prepare_datasets(self) -> Iterator[BaseDataset]:
        for files in [self.train_files, self.val_files, self.test_files]:
            yield self.dataset_class(
                files,
                self.tokenizer,
                max_length=self.max_length,
                remove_null=self.remove_null,
            )

    def train_dataloader(self, size: Optional[int] = None):
        if size is None:
            # Run training on infinite dataloader
            sampler = SWDESampler(self.data_train, replacement=True, remove_null=self.remove_null)
        else:
            # Evaluate model on training set using subset of the data
            sampler = RandomSampler(self.data_train, num_samples=size)

        return self._data_loader(self.data_train, sampler)

    def val_dataloader(self, size: Optional[int] = None):
        if size is None:
            sampler = None
        else:
            sampler = SWDESampler(self.data_val, size, replacement=True, remove_null=self.remove_null)

        return self._data_loader(self.data_val, sampler)

    def test_dataloader(self):
        return self._data_loader(self.data_test)

    def _data_loader(self, dataset: Dataset, sampler: Optional[Sampler] = None):
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True,
                          worker_init_fn=set_worker_sharing_strategy)
