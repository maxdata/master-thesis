from information_extraction.analysis.tables import get_wandb_tables, print_latex_table, aggregate_tables
from information_extraction.evaluation import get_evaluator

SWEEP_ID = 'v2h7wsaa'


def print_full_overview(evaluator, tables):
    df_agg = aggregate_tables(evaluator, tables, run_name_parts=[0])
    df_agg.index.name = 'Model'

    print_latex_table(df_agg, caption='Performance of our extraction models, compared to several strong baselines.',
                      label='table:comparison_with_baselines', highlight_axis='index')


def print_attribute_level_performance(evaluator, tables):
    df_agg = aggregate_tables(evaluator, tables, per_vertical=True, per_attribute=True, run_name_parts=[0]).loc['bert']

    table = df_agg.style.format_index(lambda s: 'NBA player' if s == 'nbaplayer' else s.capitalize(), level=0)
    table = table.format_index('\\verb|{}|', level=1)

    print_latex_table(table, 'Attribute-level performance of our best BERT configuration.',
                      'table:attribute_level_performance', long_table=True)


def main():
    tables = get_wandb_tables(SWEEP_ID)

    evaluator = get_evaluator()

    print_full_overview(evaluator, tables)
    print_attribute_level_performance(evaluator, tables)


if __name__ == '__main__':
    main()
