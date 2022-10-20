import json
from pathlib import Path


def reverse_mapping(mapping: dict):
    def reverse(sub_mapping: dict):
        return {
            open_label: closed_label
            for closed_label, open_labels in sub_mapping.items()
            if open_labels is not None
            for open_label in open_labels
        }

    return {
        vertical: {
            website: reverse(website_mapping)
            for website, website_mapping in vertical_mapping.items()
        }
        for vertical, vertical_mapping in mapping.items()
    }


with open(Path(__file__).parent / 'closed_to_open_mapping.json') as _file:
    CLOSED_TO_OPEN_MAPPING = json.load(_file)

OPEN_TO_CLOSED_MAPPING = reverse_mapping(CLOSED_TO_OPEN_MAPPING)

with open(Path(__file__).parent / 'webke_split_mapping.json') as _file:
    WEBKE_SPLIT_MAPPING = json.load(_file)
