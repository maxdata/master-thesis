from collections import namedtuple

SegmentPrediction = namedtuple('SegmentPrediction', ['batch', 'predictions', 'scores', 'embeddings'])
DocumentPrediction = namedtuple('DocumentPrediction', ['predictions', 'segments'])

EvaluationResult = namedtuple('EvaluationResult', ['metrics', 'segments', 'documents'])
