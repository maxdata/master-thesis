from collections import defaultdict
from typing import Dict

import pandas as pd

from information_extraction.analysis.tables import get_wandb_tables, print_latex_table, aggregate_tables
from information_extraction.analysis.stats import perform_aggregation_and_significance_tests
from information_extraction.evaluation import get_evaluator, Evaluator
from information_extraction.data import DOMAINS

SWEEP_ID = 'sf86fzh8'


def print_context_summary(evaluator: Evaluator, tables: Dict[str, pd.DataFrame]):
    tables = {**tables, 'context-size-Ensemble': pd.concat(combine_context_tables(tables).values())}

    df_agg = aggregate_tables(evaluator, tables, run_name_parts=[2])
    df_agg.index.name = 'Context size'

    significances = perform_aggregation_and_significance_tests(evaluator, tables, baselines=['128', '256'],
                                                               p_value=0.01, run_name_parts=[2])

    print(significances)

    df = df_agg.style.format_index(lambda s: s.upper(), axis='columns', level=1)
    print_latex_table(df, caption='Performance of a BERT model for different context sizes',
                      label='table:context_sizes', highlight_axis='index')


def print_context_performance_per_attribute(evaluator: Evaluator, tables: Dict[str, pd.DataFrame]):
    df_agg = aggregate_tables(evaluator, tables, per_vertical=True, per_attribute=True, run_name_parts=[2])
    df_agg = df_agg[('Instance', 'F1')].unstack(level=0)
    df_agg.index.names = ['Vertical', 'Attribute']
    df_agg.columns = pd.MultiIndex.from_product([['Context size'], df_agg.columns])

    table = df_agg.style.format_index(lambda s: 'NBA player' if s == 'nbaplayer' else s.capitalize(), level=0)
    table = table.format_index('\\verb|{}|', level=1)

    caption = (
        'Performance of a BERT model for different context sizes. '
        'Performance is reported using the instance-level F1 score. '
        'The best performing context size for a given attribute is marked in bold.'
    )

    print_latex_table(table, caption=caption, label='table:context_sizes_per_attribute', highlight_axis='columns',
                      long_table=True)


def combine_context_tables(tables: Dict[str, pd.DataFrame]):
    per_vertical = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run_name, table in tables.items():
        _, _, context_size, vertical = run_name.split('-')

        for _, row in table.iterrows():
            for attribute in DOMAINS[vertical]:
                per_vertical[vertical][row['doc_id']][attribute].append({
                    field: row[f'{attribute}/{field}']
                    for field in ['true', 'pred', 'conf', 'f1', 'em']
                })

    combined_tables = {}
    for vertical, entries in per_vertical.items():
        rows = []

        for doc_id, doc_entries in entries.items():
            new_row = {'doc_id': doc_id}

            for attribute, attribute_values in doc_entries.items():
                best_prediction = max(attribute_values, key=lambda fields: fields['conf'])
                new_row |= {
                    f'{attribute}/{field}': value
                    for field, value in best_prediction.items()
                }

            rows.append(new_row)

        combined_tables[vertical] = pd.DataFrame(rows)

    return combined_tables


def main():
    tables = get_wandb_tables(SWEEP_ID)

    evaluator = get_evaluator()

    print_context_summary(evaluator, tables)
    print_context_performance_per_attribute(evaluator, tables)


if __name__ == '__main__':
    main()
