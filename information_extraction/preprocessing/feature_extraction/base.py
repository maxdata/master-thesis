from abc import ABC, abstractmethod
from io import StringIO
from multiprocessing import Process, Queue
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple

from lxml import etree
import lxml.html
from lxml.html.clean import Cleaner

import pandas as pd

from transformers import PreTrainedTokenizer
from transformers.data.metrics.squad_metrics import normalize_answer

from information_extraction.data import DOMAINS
from information_extraction.data.metrics import normalize_answer
from information_extraction.data.ground_truths import GroundTruths
from information_extraction.util.multiprocessing import status_bar


# We need some extra tokens to encode the attribute type and special tokens
EXTRA_TOKEN_SPACE = 10


class BaseExtractor(ABC):
    output_format = '{split}/{vertical}/{website}/features-{max_length}.csv'
    position_attributes = {'x', 'y', 'width', 'height'}

    def __init__(self, input_path: Path, output_path: Path, tokenizer: PreTrainedTokenizer,
                 max_length: int, domains: Optional[dict] = None, include_positions: bool = False,
                 num_workers: int = 1):
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domains = domains or DOMAINS
        self.num_workers = num_workers if num_workers > 0 else None

        self.num_too_large = 0
        self.ground_truths = GroundTruths(input_path / 'groundtruth')

        if not include_positions:
            self.position_attributes = set()

    @abstractmethod
    def feature_representation(self, elem: etree.Element) -> str:
        raise NotImplementedError

    @staticmethod
    def text_representation(elem: etree.Element) -> str:
        return re.sub(r'\s+', ' ', ' '.join(elem.itertext())).strip()

    def process_dataset(self) -> None:
        for split in ['train', 'val', 'test']:
            for vertical in os.listdir(self.input_path / split):
                for website in os.listdir(self.input_path / split / vertical):
                    self.process_directory(split, vertical, website)

    def process_directory(self, split: str, vertical: str, website: str) -> Path:
        base_dir = self.input_path / split / vertical / website
        files = [
            filename
            for filename in os.listdir(base_dir)
            if filename.endswith('.htm')
        ]

        self.num_too_large = 0

        extracted_features = self.get_features_from_files(f'{split}/{vertical}-{website}-{self.max_length}', [
            (str(base_dir / filename), f'{vertical}/{website}/{filename[:-4]}')
            for filename in files
        ])
        df_features = pd.DataFrame(extracted_features)

        destination = self.output_path / self.output_format.format(
            split=split,
            vertical=vertical,
            website=website,
            max_length=self.max_length,
        )
        destination.parent.mkdir(exist_ok=True, parents=True)

        df_features.to_csv(destination, index=False)

        if self.num_too_large > 0:
            num_total = self.num_too_large + len(df_features)
            print(f'Warning: {vertical}-{website} has {self.num_too_large}/{num_total} '
                  f'({self.num_too_large / num_total * 100:.2f}%) unsplittable texts that are '
                  f'longer than {self.max_length} tokens')

        return destination

    def get_features_from_files(self, label: str, files: List[Tuple[str, str]]):
        task_queue = Queue()
        stat_queue = Queue()
        result_queue = Queue()

        for entry in files:
            task_queue.put(entry)

        for _ in range(self.num_workers):
            task_queue.put(None)

        processes = [ExtractorProcess(task_queue, stat_queue, result_queue, self) for _ in range(self.num_workers)]
        processes.append(Process(target=status_bar, args=(stat_queue, len(files), label)))

        for process in processes:
            process.start()

        extracted_features = []
        num_finished = 0

        while num_finished < self.num_workers:
            features = result_queue.get()
            if features is None:
                num_finished += 1
            else:
                extracted_features.extend(features)

        return extracted_features

    def get_features_from_file(self, filename: str, ground_truth: dict) -> List[dict]:
        with open(filename) as _file:
            html = _file.read()

        cleaned_html = self.clean_html(html)

        tree = lxml.html.parse(StringIO(cleaned_html))

        safe_attrs = Cleaner.safe_attrs | {f'data-{attr}' for attr in self.position_attributes}
        cleaner = Cleaner(style=True, safe_attrs=safe_attrs)

        cleaned_body = cleaner.clean_html(tree.find('body'))

        return self.extract_features(cleaned_body, ground_truth)

    def extract_features(self, elem: etree.Element, ground_truth: dict) -> List[dict]:
        representation = self.feature_representation(elem).strip()

        if not representation:
            return []

        if len(self.tokenizer.tokenize(representation)) > self.max_length - EXTRA_TOKEN_SPACE:
            if not len(elem):
                # This element is too large to fit in the document, but cannot
                # be split into sub-elements
                self.num_too_large += 1
                return []

            # We split the element that is too large into its children
            features = []
            for child in elem:
                features.extend(self.extract_features(child, ground_truth))

            return features

        normalized_text = normalize_answer(self.text_representation(elem))

        results = [{
            'text': representation,
            **{
                f'pos/{attr}': elem.get(f'data-{attr}')
                for attr in self.position_attributes
            },
        }]
        for attribute, possible_values in ground_truth.items():
            found_values = []
            for value in possible_values:
                pattern = re.compile(f'\\b{re.escape(normalize_answer(value))}\\b')

                if value != '<NULL>' and pattern.search(normalized_text):
                    found_values.append(value)

            if found_values:
                results = [
                    {**result, attribute: value}
                    for value in found_values
                    for result in results
                ]
            else:
                results = [
                    {**result, attribute: None}
                    for result in results
                ]

        return results

    @staticmethod
    def clean_html(html: str) -> str:
        # https://stackoverflow.com/a/25920392
        html = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', html)

        # lxml automatically converts HTML entities, which we do not want
        html = html.replace('&', '&amp;')

        return html


class ExtractorProcess(Process):
    def __init__(self, task_queue: Queue, stat_queue: Queue, output_queue: Queue, extractor: BaseExtractor):
        super().__init__()

        self.task_queue = task_queue
        self.stat_queue = stat_queue
        self.output_queue = output_queue
        self.extractor = extractor

    def run(self) -> None:
        while True:
            next_task = self.task_queue.get()

            if next_task is None:
                self.output_queue.put(None)
                break

            filename, doc_id = next_task

            try:
                features = self.extractor.get_features_from_file(filename, self.extractor.ground_truths[doc_id])

                for feature in features:
                    feature['doc_id'] = doc_id

                self.output_queue.put(features)
                self.stat_queue.put('success')
            except Exception as e:
                self.stat_queue.put(f'Error for {next_task}: {e}')

        return
