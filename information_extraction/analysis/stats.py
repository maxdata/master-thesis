from typing import Dict, List, Optional, Union

import pandas as pd
from scipy.stats import wilcoxon

from information_extraction.analysis.tables import aggregate_tables
from information_extraction.evaluation import Evaluator


def perform_significance_test(df: pd.DataFrame, baseline: str, p_value: Optional[float] = None,
                              group_by: Optional[List[str]] = None):
    if group_by is not None:
        return pd.concat({
            group_name: perform_significance_test(df_group, baseline, p_value=p_value)
            for group_name, df_group in df.groupby(level=group_by)
        })

    rows = []

    for key in df.index.get_level_values(0).unique():
        if key == baseline:
            rows.append(pd.Series([1] * len(df.columns), index=df.columns, name=baseline))
            continue

        result = []

        for column in df.columns:
            baseline_results = df.loc[baseline, column]
            other_results = df.loc[key, column]

            result.append(wilcoxon(baseline_results, other_results, alternative='less').pvalue)

        rows.append(pd.Series(result, index=df.columns, name=key))

    p_values = pd.DataFrame(rows)

    return p_values if p_value is None else p_values < p_value


def perform_aggregation_and_significance_tests(evaluator: Evaluator, tables: Dict[str, pd.DataFrame],
                                               baselines: Union[str, List[str]], p_value: Optional[float] = None,
                                               **agg_kwargs):
    df_global = aggregate_tables(evaluator, tables, per_document=True, **agg_kwargs)['Global']
    df_instance = aggregate_tables(evaluator, tables, per_document=True, per_attribute=True, **agg_kwargs)['Instance']

    df_global.to_csv('global.csv')
    df_instance.to_csv('instance.csv')

    dfs = {
        'Global': (df_global, [level for level in df_global.index.names[1:]
                               if level not in ['doc_id']] or None),
        'Instance': (df_instance, [level for level in df_instance.index.names[1:]
                                   if level not in ['doc_id', 'attribute']] or None),
    }

    if not isinstance(baselines, list):
        baselines = [baselines]

    results = {
        baseline: pd.concat(
            {
                metric_group: perform_significance_test(
                    metric_df, baseline, p_value, group_by=group_by
                )
                for metric_group, (metric_df, group_by) in dfs.items()
            },
            axis=1,
        )
        for baseline in baselines
    }
    return pd.concat(results)
