from lxml import etree
from pathlib import Path
import re
from typing import Iterable, List, Optional

from transformers import PreTrainedTokenizer

from .base import BaseExtractor


class SimpleHtmlExtractor(BaseExtractor):
    def get_representation_parts(self, elem: etree.Element, top_level: bool = False) -> List[str]:
        if elem is None:
            return []

        result = [f'<{elem.tag}>']

        if (value := elem.get('id')) is not None:
            result.append(f'id={value.strip()}')

        if (values := elem.get('class')) is not None:
            result.extend(f'class={value.strip()}' for value in values.split())

        if elem.text is not None:
            result.append(elem.text.strip())

        for child in elem:
            result.extend(self.get_representation_parts(child))

        if elem.tail is not None and not top_level:
            result.append(elem.tail.strip())

        result.append(f'</{elem.tag}>')

        return result

    def feature_representation(self, elem: etree.Element) -> str:
        return ' '.join(self.get_representation_parts(elem, top_level=True))


class HtmlExtractor(BaseExtractor):
    parent_prefix = 'p'
    tag_suffix = 't'
    id_suffix = 'i'
    class_suffix = 'c'

    def __init__(self, input_path: Path, output_path: Path, tokenizer: PreTrainedTokenizer,
                 max_length: int, parent_depth: int = 1, encode_class: bool = True, encode_id: bool = True,
                 encode_tag_subset: Optional[Iterable[str]] = None, split_attributes: bool = False,
                 domains: Optional[dict] = None, include_positions: bool = False, num_workers: int = 1):
        super().__init__(input_path, output_path, tokenizer, max_length, domains, include_positions, num_workers)
        self.parent_depth = parent_depth
        self.encode_class = encode_class
        self.encode_id = encode_id
        self.encode_tag_subset = encode_tag_subset
        self.split_attributes = split_attributes

    def get_ancestor_encoding(self, elem: etree.Element, depth: int = 0) -> List[str]:
        if depth >= self.parent_depth or elem is None:
            return []
        ancestor_encoding = self.get_ancestor_encoding(elem.getparent(), depth=depth + 1)

        current_parent = [
            f'{self.parent_prefix}{depth}{self.tag_suffix} {elem.tag}',
        ]

        if self.encode_tag_subset is None or elem.tag in self.encode_tag_subset:
            if (value := elem.get('id')) is not None and self.encode_id:
                parts = self.split_attribute_value(value) if self.split_attributes else [value]
                current_parent.extend(f'{self.parent_prefix}{depth}{self.id_suffix} {part}' for part in parts)

            if (value := elem.get('class')) is not None and self.encode_class:
                if self.split_attributes:
                    parts = self.split_attribute_value(value)
                else:
                    parts = value.split()
                current_parent.extend(f'{self.parent_prefix}{depth}{self.class_suffix} {part}' for part in parts)

        ancestor_encoding.extend(current_parent)

        return ancestor_encoding

    def feature_representation(self, elem: etree.Element) -> str:
        texts = []

        for text in elem.xpath(".//text()"):
            if not text.strip():
                continue

            parent = text.getparent() if text.is_text else text.getparent().getparent()
            prefix = ' '.join(self.get_ancestor_encoding(parent))
            texts.append(f'{prefix} {text.strip()}')

        return ' '.join(texts)

    @staticmethod
    def split_attribute_value(value: str) -> List[str]:
        def split_dashes(s: str) -> List[str]:
            return re.split(r'[_\- ]', s)

        def split_camel(s: str) -> List[str]:
            parts = []
            current = []

            for i in range(len(s) - 1):
                current.append(s[i])
                if s[i].islower() and s[i + 1].isupper():
                    parts.append(''.join(current))
                    current = []

            current.append(s[-1])
            parts.append(''.join(current))

            return parts

        parts = {value}

        for transform in (split_dashes, split_camel):
            parts = {y for x in parts for y in transform(x) if y}

        return list(parts)
