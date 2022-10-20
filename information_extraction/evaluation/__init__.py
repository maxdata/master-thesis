from information_extraction.data.metrics import compute_f1, compute_exact
from .evaluator import Evaluator


def get_evaluator():
    return Evaluator({'f1': compute_f1, 'em': compute_exact})
