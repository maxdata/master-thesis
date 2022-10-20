from pathlib import Path
from typing import Optional

from information_extraction.dtypes import EvaluationResult, PathLike
from .base import BaseCallback


class SavePredictions(BaseCallback):
    def __init__(self, segments_dir: Optional[PathLike] = None, documents_dir: Optional[PathLike] = None):
        self.segments_dir = Path(segments_dir) if segments_dir is not None else None
        self.documents_dir = Path(documents_dir) if documents_dir is not None else None

    def on_evaluation_end(self, identifier: str, results: EvaluationResult):
        if results.segments is not None and self.segments_dir is not None:
            self.segments_dir.mkdir(exist_ok=True, parents=True)
            results.segments.to_csv(self.segments_dir / f'segments-{identifier}.csv', index=False)

        if results.documents is not None and self.documents_dir is not None:
            self.documents_dir.mkdir(exist_ok=True, parents=True)
            results.documents.to_csv(self.documents_dir / f'documents-{identifier}.csv', index=False)
