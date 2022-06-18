from collections import defaultdict
from pathlib import Path
from typing import Union


class GroundTruths:
    def __init__(self, ground_truth_dir: Union[str, Path]):
        self.ground_truths = self.read_ground_truths(Path(ground_truth_dir))

    def __getitem__(self, item: str):
        result = self.ground_truths

        for value in item.split('/'):
            result = result[value]

        return result

    def read_ground_truths(self, ground_truth_dir: Path):
        ground_truths = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for ground_truth_file in ground_truth_dir.glob('*/*.txt'):
            with open(ground_truth_file) as _file:
                input_data = _file.read()

            domain, website, attribute = ground_truth_file.stem.split('-')

            for doc_id, values in self.parse_ground_truth_file(input_data).items():
                ground_truths[domain][website][doc_id][attribute] = values

        return ground_truths

    @staticmethod
    def parse_ground_truth_file(input_data: str):
        entries = input_data.splitlines()[2:]

        data = {}

        for entry in entries:
            parts = entry.split('\t')
            data[parts[0]] = [
                '' if value == '<NULL>' else value
                for value in parts[2:]
            ]

        return data
