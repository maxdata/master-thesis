from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler
from transformers import BertTokenizerFast, T5TokenizerFast

from data.base import BaseDataset
from data.bert import BertDataset
from data.sampler import DocumentSampler, SegmentSampler
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
            base_sampler = SegmentSampler(self.data_train, replacement=True, remove_null=self.remove_null)
        else:
            # Evaluate model on training set using subset of the data
            base_sampler = RandomSampler(self.data_train, num_samples=size)

        return self._segment_loader(self.data_train, base_sampler)

    def val_dataloader(self, size: Optional[int] = None):
        if size is None:
            base_sampler = None
        else:
            base_sampler = SegmentSampler(self.data_val, size, replacement=True, remove_null=self.remove_null)

        return self._segment_loader(self.data_val, base_sampler)

    def test_dataloader(self):
        return self._segment_loader(self.data_test)

    def train_document_dataloaders(self, num_documents: Optional[int] = None):
        if num_documents is None:
            sampler = DocumentSampler(self.data_train, shuffle=True, replacement=True, batch_size=self.batch_size)
        else:
            sampler = DocumentSampler(self.data_train, num_documents=num_documents, batch_size=self.batch_size)

        for doc_id, document_indices in sampler:
            yield doc_id, self._document_loader(self.data_train, document_indices)

    def val_document_dataloaders(self, num_documents: Optional[int] = None):
        sampler = DocumentSampler(self.data_train, num_documents=num_documents, batch_size=self.batch_size)

        for doc_id, document_indices in sampler:
            yield doc_id, self._document_loader(self.data_train, document_indices)

    def test_document_dataloaders(self):
        sampler = DocumentSampler(self.data_train, batch_size=self.batch_size)

        for doc_id, document_indices in sampler:
            yield doc_id, self._document_loader(self.data_train, document_indices)

    def _segment_loader(self, dataset: BaseDataset, base_sampler: Optional[Sampler] = None):
        if base_sampler is None:
            base_sampler = SequentialSampler(dataset)

        sampler = BatchSampler(base_sampler, batch_size=self.batch_size, drop_last=False)

        return self._data_loader(dataset, sampler)

    def _document_loader(self, dataset: BaseDataset, iterator: Iterable[Tuple[str, Iterable[int]]]):
        for attribute, indices in iterator:
            yield attribute, self._data_loader(dataset, indices)


    def _data_loader(self, dataset: BaseDataset, sampler: Union[Sampler, Iterable, None]):
        return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True, worker_init_fn=set_worker_sharing_strategy)
