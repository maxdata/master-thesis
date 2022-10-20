from typing import List, Tuple

from lxml import etree

from .base import BaseExtractor


class TextExtractor(BaseExtractor):
    def feature_representation(self, elem: etree.Element) -> Tuple[List[str], List[str]]:
        texts = self.text_representation(elem).split()
        ancestors = ['' for _ in range(len(texts))]

        return texts, ancestors
