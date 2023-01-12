from typing import Dict, List, Tuple

from information_extraction.data.metrics import normalize_answer


def get_precision_recall_f1(y_true: Dict[str, List[Tuple[str, str]]],
                            y_pred: Dict[str, List[Tuple[str, str]]]) -> Tuple[float, float, float]:
    tp = tpfp = tpfn = 0

    for doc_id, predictions in y_pred.items():
        a_pred = {(label, normalize_answer(value)) for label, value in predictions}
        a_true = {(label, normalize_answer(value)) for label, value in y_true[doc_id]}

        tp += len(a_true & a_pred)
        tpfp += len(a_pred)
        tpfn += len(a_true)

    precision = tp / tpfp if tpfp > 0 else 0
    recall = tp / tpfn if tpfn > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1
