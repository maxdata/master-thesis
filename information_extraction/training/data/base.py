from abc import ABC, abstractmethod
import json
from pathlib import Path
import time
from typing import Iterable, Iterator, List, Optional, Tuple, Union

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer


class BaseDataset(Dataset, ABC):
    docs: List[str] = []
    inputs: List[str] = []
    ancestors: List[List[Union[str, Optional[int]]]] = []
    targets: List[str] = []
    features: List[str] = []

    indices_with_data: List[int] = []
    indices_without_data: List[int] = []

    def __init__(self, files: Iterable[Union[str, Path]], tokenizer: PreTrainedTokenizer,
                 remove_null: bool = False, max_length: Optional[int] = None):
        self.files = list(map(Path, files))
        self.tokenizer = tokenizer
        self.remove_null = remove_null

        self.tokenize_kwargs = {'return_tensors': 'pt', 'padding': True, 'is_split_into_words': True}
        if max_length is not None:
            self.tokenize_kwargs['truncation'] = True
            self.tokenize_kwargs['max_length'] = max_length

        if self.files:
            print(f'Loading {self.__class__.__name__}...', end=' ', flush=True)
            start = time.time()

            self.docs, self.inputs, self.ancestors, self.targets, self.features = self.read_data()
            self.prepare_inputs()

            self.indices_with_data = [i for i, t in enumerate(self.targets) if t]
            self.indices_without_data = [i for i, t in enumerate(self.targets) if not t]

            end = time.time()
            print(f'done in {end - start:.1f}s')

    def prepare_inputs(self):
        pass

    def read_data(self):
        return [list(items) for items in zip(*(
            sample
            for file in self.files
            for sample in self.read_csv(file)
        ))]

    def read_csv(self, file: Path) -> Iterator[Tuple[str, ...]]:
        with open(file) as _file:
            data = json.load(_file)

        for entry in data:
            features = [key
                        for key in entry
                        if key not in ('doc_id', 'text', 'ancestors')
                        and not key.startswith('pos/')]

            for feature in features:
                if entry[feature] or not self.remove_null:
                    yield entry['doc_id'], entry['text'], entry['ancestors'], entry[feature] or '', feature

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
