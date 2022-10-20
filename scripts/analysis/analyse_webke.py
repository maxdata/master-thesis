from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from information_extraction.analysis.tables import get_wandb_tables, get_wandb_artifact, print_latex_table, aggregate_tables
from information_extraction.analysis.stats import perform_significance_test, perform_aggregation_and_significance_tests
from information_extraction.evaluation import get_evaluator, Evaluator
from information_extraction.evaluation.openie import get_precision_recall_f1
from information_extraction.data.ground_truths import GROUND_TRUTHS
from information_extraction.data.metrics import compute_f1

from information_extraction.analysis.mappings import CLOSED_TO_OPEN_MAPPING, OPEN_TO_CLOSED_MAPPING, WEBKE_SPLIT_MAPPING
from information_extraction.config import DATA_DIR


SWEEP_ID = '3sn14gvv'

EXPANDED_SWDE_DIR = DATA_DIR / 'ExpandedSWDE'


def get_webke_results(split_name: str = 'test') -> dict:
    webke_result_dir = get_wandb_artifact('webke-predictions')

    results = {}

    for vertical in os.listdir(webke_result_dir):
        results[vertical] = {}
        for filename in os.listdir(webke_result_dir / vertical / split_name):
            with open(webke_result_dir / vertical / split_name / filename) as _file:
                data = json.load(_file)['pred_list_pred']

            doc_id = filename.split('.')[0]
            original_website, original_doc_id = WEBKE_SPLIT_MAPPING[split_name][vertical].get(doc_id, ('webke', doc_id))

            if original_website == 'webke':
                # We could not map this back to a document
                continue

            original_website = original_website.split('(')[0].split('-')[-1]

            doc_id = f'{vertical}/{original_website}/{original_doc_id}'

            results[vertical][doc_id] = data

    return results


def map_webke_to_closed(data: List[Tuple[str, str]], mapping: Dict[str, str], ground_truths: Dict[str, str]):
    values = defaultdict(list)

    for open_label, value in data:
        if open_label in mapping:
            values[mapping[open_label]].append(value)

    result = {closed_label: {'prediction': '', 'confidence': 1} for closed_label in ground_truths}

    for closed_label, predictions in values.items():
        result[closed_label] = {
            'prediction': max([
                (compute_f1(a_true, a_pred), a_pred)
                for a_true in ground_truths[closed_label]
                for a_pred in predictions
            ])[1],
            'confidence': 1,
        }

    return result


def get_webke_as_closed(evaluator: Evaluator):
    webke_predictions = get_webke_results()

    closed_results = {}

    for vertical, predictions in webke_predictions.items():
        vertical_predictions = {}
        for doc_id, prediction in predictions.items():
            _, original_website, _ = doc_id.split('/')
            vertical_predictions[doc_id] = map_webke_to_closed(
                prediction,
                OPEN_TO_CLOSED_MAPPING[vertical][original_website],
                GROUND_TRUTHS[doc_id]
            )

        closed_results[vertical] = evaluator.documents_to_table(vertical_predictions)

    return closed_results


def get_own_as_closed(evaluator: Evaluator, sweep_id: str):
    tables = get_wandb_tables(sweep_id)

    return {
        run_name: evaluator.documents_to_table({
            row['doc_id']: {
                attribute: {
                    'prediction': row[f'{attribute}/pred'],
                    'confidence': row[f'{attribute}/conf']
                }
                for attribute in GROUND_TRUTHS[row['doc_id']]
            }
            for _, row in table.iterrows()
        })
        for run_name, table in tables.items()
    }


def print_closed_performance():
    valid_rows = [
        {
            'vertical': vertical,
            'website': website,
            'attribute': attribute,
        }
        for vertical, vertical_mapping in CLOSED_TO_OPEN_MAPPING.items()
        for website, website_mapping in vertical_mapping.items()
        for attribute, open_labels in website_mapping.items()
        if open_labels is not None
    ]

    evaluator = get_evaluator()

    results = {
        'Ours': get_own_as_closed(evaluator, SWEEP_ID),
        'WebKE': get_webke_as_closed(evaluator),
    }

    tables = {
        f'{model}-{run_name}': table
        for model, runs in results.items()
        for run_name, table in runs.items()
    }

    df_agg = aggregate_tables(evaluator, tables, per_vertical=True, run_name_parts=[0], subset=valid_rows)
    df_agg = df_agg.swaplevel().sort_index(ascending=[True, False])
    df_agg.index.names = ['Vertical', 'Model']

    significances = perform_aggregation_and_significance_tests(evaluator, tables, baselines=['WebKE', 'Ours'],
                                                               p_value=0.01, run_name_parts=[0], per_vertical=True,
                                                               subset=valid_rows)

    print(significances.sort_index(ascending=[False, True, False]))

    print_latex_table(df_agg.style.format_index('\\texttt{{{}}}', level=0),
                      'Comparison between our own model and WebKE on the regular SWDE dataset.',
                      'table:webke_comparison_swde', highlight_axis='index')


def limit_to_closed(predictions: Dict[str, List[Tuple[str, str]]]):
    new_predictions = {}

    for doc_id, y_pred in predictions.items():
        vertical, website, _ = doc_id.split('/')
        mapping = OPEN_TO_CLOSED_MAPPING[vertical][website]

        new_predictions[doc_id] = [
            (mapping[label], value) for label, value in y_pred if label in mapping
        ]

    return new_predictions


def get_webke_as_open():
    results = get_webke_results()

    return {vertical: limit_to_closed(data) for vertical, data in results.items()}


def get_own_as_open(sweep_id: str):
    tables = get_wandb_tables(sweep_id)

    allowed_docs = set(
        f'{vertical}/{website.split("-")[1].split("(")[0]}/{doc_id}'
        for vertical, mapping in WEBKE_SPLIT_MAPPING['test'].items()
        for website, doc_id in mapping.values()
    )

    results = defaultdict(dict)

    for table in tables.values():
        for _, row in table.iterrows():
            if row['doc_id'] not in allowed_docs:
                continue

            vertical, website, _ = row['doc_id'].split('/')
            mapping = CLOSED_TO_OPEN_MAPPING[vertical][website]

            results[vertical][row['doc_id']] = [
                (attribute, row[f'{attribute}/pred'])
                for attribute, open_labels in mapping.items()
                if row[f'{attribute}/pred'] and open_labels is not None
            ]

    return results


def get_expanded_swde(expanded_swde_dir: Path):
    ground_truths = defaultdict(list)

    for vertical in os.listdir(expanded_swde_dir):
        for filename in os.listdir(expanded_swde_dir / vertical):
            with open(expanded_swde_dir / vertical / filename) as _file:
                data = json.load(_file)

            website = filename.split('(')[0].split('-')[-1]

            for identifier, entries in data.items():
                doc_id = f'{vertical}/{website}/{identifier[:-4]}'

                for label, values in entries.items():
                    ground_truths[doc_id].extend((label, value) for value in values)

    return limit_to_closed(ground_truths)


def print_open_performance():
    results = {
        'Ours': get_own_as_open(SWEEP_ID),
        'WebKE': get_webke_as_open(),
    }

    ground_truth = get_expanded_swde(EXPANDED_SWDE_DIR)

    df_agg = pd.concat({
        key: pd.DataFrame({
            vertical: pd.Series(get_precision_recall_f1(ground_truth, predictions), index=['P', 'R', 'F1'])
            for vertical, predictions in data.items()
        }).T
        for key, data in results.items()
    }).swaplevel().sort_index(ascending=[True, False])

    df_per_doc = pd.concat({
        key: pd.DataFrame({
            (vertical, doc_id): pd.Series(get_precision_recall_f1(ground_truth, {doc_id: doc_predictions}),
                                          index=['P', 'R', 'F1'])
            for vertical, predictions in data.items()
            for doc_id, doc_predictions in predictions.items()
        }).T
        for key, data in results.items()
    }).sort_index()
    df_per_doc.index.names = ['Model', 'vertical', 'doc_id']

    significances = pd.concat({
        baseline: perform_significance_test(df_per_doc, baseline, p_value=0.01, group_by=['vertical'])
        for baseline in ['WebKE', 'Ours']
    })

    print(significances.sort_index(ascending=[False, True, False]))

    print_latex_table(df_agg.style.format_index('\\texttt{{{}}}', level=0),
                      'Comparison between our own model and WebKE on the Expanded SWDE dataset.',
                      'table:webke_comparison_expanded_swde',
                      highlight_axis='index')


def main():
    print_closed_performance()
    print_open_performance()


if __name__ == '__main__':
    main()
